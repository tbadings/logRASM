from functools import partial

import jax
import numpy as np
from flax.training.train_state import TrainState
from jax import numpy as jnp

from core.commons import MultiRectangularSet
from core.jax_utils import lipschitz_coeff
from core.plot import plot_dataset


class Learner:
    '''
    Main learner class.

    '''

    def __init__(self, env, args):
        '''
        Initialize the learner.

        :param env: Environment. 
        :param args: Command line arguments given. 
        '''

        self.env = env
        self.linfty = False  # L_infty has only experimental support (not used in experiments)

        # Copy some arguments
        self.auxiliary_loss = args.auxiliary_loss
        self.lambda_lipschitz = args.loss_lipschitz_lambda  # Lipschitz factor
        self.max_lip_certificate = args.loss_lipschitz_certificate  # Above this value, incur loss
        self.max_lip_policy = args.loss_lipschitz_policy  # Above this value, incur loss
        self.weighted = args.weighted
        self.cplip = args.cplip
        self.split_lip = args.split_lip
        self.min_lip_policy = args.min_lip_policy_loss
        self.exp_certificate = args.exp_certificate
        self.loss_decr_squared = args.loss_decr_squared
        self.loss_decr_max = args.loss_decr_max
        self.EPS_decrease = args.eps_decrease

        # Set batch sizes
        self.batch_size_total = int(args.batch_size)
        self.batch_size_base = int(args.batch_size * (1 - args.counterx_fraction))
        self.batch_size_counterx = int(args.batch_size * args.counterx_fraction)

        # Calculate the number of samples for each region type (without counterexamples)
        MIN_SAMPLES = max(int(args.min_fraction_samples_per_region * self.batch_size_base), 1)

        totvol = env.state_space.volume
        if isinstance(env.init_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.init_space.sets])
            self.num_samples_init = tuple(np.maximum(np.ceil(rel_vols * self.batch_size_base), MIN_SAMPLES).astype(int))
        else:
            self.num_samples_init = np.maximum(MIN_SAMPLES,
                                               np.ceil(env.init_space.volume / totvol * self.batch_size_base)).astype(
                int)
        if isinstance(env.unsafe_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.unsafe_space.sets])
            self.num_samples_unsafe = tuple(
                np.maximum(MIN_SAMPLES, np.ceil(rel_vols * self.batch_size_base)).astype(int))
        else:
            self.num_samples_unsafe = np.maximum(np.ceil(env.unsafe_space.volume / totvol * self.batch_size_base),
                                                 MIN_SAMPLES).astype(int)
        if isinstance(env.target_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.target_space.sets])
            self.num_samples_target = tuple(
                np.maximum(np.ceil(rel_vols * self.batch_size_base), MIN_SAMPLES).astype(int))
        else:
            self.num_samples_target = np.maximum(MIN_SAMPLES, np.ceil(
                env.target_space.volume / totvol * self.batch_size_base)).astype(int)

        # Infer the number of expected decrease samples based on the other batch sizes
        self.num_samples_decrease = np.maximum(self.batch_size_base
                                               - np.sum(self.num_samples_init)
                                               - np.sum(self.num_samples_unsafe)
                                               - np.sum(self.num_samples_target), 1).astype(int)

        if not args.silent:
            print(f'- Num. base train samples per batch: {self.batch_size_base}')
            print(f'-- Initial state: {self.num_samples_init}')
            print(f'-- Unsafe state: {self.num_samples_unsafe}')
            print(f'-- Target state: {self.num_samples_target}')
            print(f'-- Expected decrease: {self.num_samples_decrease}')
            print(f'- Num. counterexamples per batch: {self.batch_size_counterx}\n')

        if self.lambda_lipschitz > 0 and not args.silent:
            print('- Learner setting: Enable Lipschitz loss')
            print(f'--- For certificate up to: {self.max_lip_certificate:.3f}')
            print(f'--- For policy up to: {self.max_lip_policy:.3f}')

        self.glob_min = 0.1
        self.N_expectation = args.learner_N_expectation  # Number of samples to approximate expectation

        # Define vectorized functions for loss computation
        self.loss_exp_decrease_vmap = jax.vmap(self.loss_exp_decrease, in_axes=(None, None, 0, 0, 0, None), out_axes=0)

        return

    def loss_exp_decrease(self, V_state, V_params, x, u, noise_key, probability_bound):
        '''
        Compute expected certificate value in the new state for the loss related to condition 3 (expected decrease).
        
        :param V_state: Certificate neural network. 
        :param V_params: Parameters of the certificate neural network. 
        :param x: State.
        :param u: Action.
        :param noise_key: key of the random number generator.
        :param probability_bound: The probability bound of the specification that we aim to certify.
        :return: Expected certificate value in the new state.
        '''

        # For each given noise_key, compute the successor state for the pair (x,u)
        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)

        # Function apply_fn does a forward pass in the certificate network for all successor states in state_new,
        # which approximates the value of the certificate for the successor state (using different noise values).
        # Then, the loss term is zero if the expected decrease in certificate value is at least tau*K.
        if self.exp_certificate:
            V_expected = jnp.log(jnp.mean(jnp.exp(
                jnp.minimum(V_state.apply_fn(V_params, state_new), jnp.log(2) - jnp.log(1 - probability_bound))
            )))
            # The jnp.minimum ensures that values do not become infinite (due to the exponential).
            # Note that this retains the soundness since we may cap any valid lograsm at - jnp.log(1 - probability_bound).
        else:
            V_expected = jnp.mean(
                jnp.minimum(V_state.apply_fn(V_params, state_new), 2 / (1 - probability_bound))
            )

        return V_expected

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self,
                   key: jax.Array,
                   V_state: TrainState,
                   Policy_state: TrainState,
                   counterexamples,
                   mesh_loss,
                   probability_bound,
                   expDecr_multiplier
                   ):
        '''
        Perform one step of training the neural network.

        :param key: key of the random number generator.
        :param V_state: Certificate network.
        :param Policy_state: Policy network.
        :param counterexamples: Current list of counterexamples.
        :param mesh_loss: float, determining the largest mesh for which a loss of 0 implies that the condition is satisfied. 
        :param probability_bound: The probability bound of the specification that we aim to certify. 
        :param expDecr_multiplier: Multiplier of the expected decrease loss. 
        :return:
           - V_grads: Gradients of the certificate network. 
           - Policy_grads: Gradients of the policy network. 
           - infos: Dictionary, giving the total loss as well as each component (total, init, unsafe, expDecrease_mean, expDecrease_max (not used), Lipschitz (not used)).
           - key: key of the random number generator.
           - samples_in_batch: Dictionary, giving the samples used in the batch per category. 
        '''

        # Generate all random keys
        key, cx_key, init_key, unsafe_key, target_key, decrease_key, noise_key, perturbation_key = \
            jax.random.split(key, 8)

        # Sample from the full list of counterexamples
        if len(counterexamples) > 0:
            # Randomly sample counterexamples from the buffer
            cx = jax.random.choice(cx_key, counterexamples, shape=(self.batch_size_counterx,), replace=False)
            cx_samples = cx[:, :-3]

            # Determine which counterexamples belong to which categories
            cx_bool_init = cx[:, -2] > 0
            cx_bool_unsafe = cx[:, -1] > 0
            cx_bool_decrease = cx[:, -3] > 0
        else:
            # No counterexamples in the buffer yet (e.g., in first iteration)
            cx_samples = cx_bool_init = cx_bool_unsafe = cx_bool_decrease = False

        # Sample from each region of interest
        samples_init = self.env.init_space.sample(rng=init_key, N=self.num_samples_init)
        samples_unsafe = self.env.unsafe_space.sample(rng=unsafe_key, N=self.num_samples_unsafe)
        samples_target = self.env.target_space.sample(rng=target_key, N=self.num_samples_target)
        samples_decrease = self.env.state_space.sample(rng=decrease_key, N=self.num_samples_decrease)

        # Exclude samples from target set
        samples_decrease_bool_not_targetUnsafe = self.env.target_space.jax_not_contains(samples_decrease)

        def loss_fun(certificate_params, policy_params):

            # Small epsilon used in the initial/unsafe loss terms
            EPS_init = 0.1
            EPS_unsafe = 0.1
            EPS_decrease = self.EPS_decrease

            # Compute Lipschitz coefficients.
            lip_certificate, _ = lipschitz_coeff(certificate_params, self.weighted, self.cplip, self.linfty)
            lip_policy, _ = lipschitz_coeff(policy_params, self.weighted, self.cplip, self.linfty)
            lip_policy = jnp.maximum(lip_policy, self.min_lip_policy)

            # Calculate K factor
            if self.linfty and self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_linfty_A + self.env.lipschitz_f_linfty_B * lip_policy)
            elif self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_l1_A + self.env.lipschitz_f_l1_B * lip_policy)
            elif self.linfty:
                K = lip_certificate * (self.env.lipschitz_f_linfty * (lip_policy + 1))
            else:
                K = lip_certificate * (self.env.lipschitz_f_l1 * (lip_policy + 1))

            #####

            # Compute certificate values in each of the relevant state sets
            V_init = jnp.ravel(V_state.apply_fn(certificate_params, samples_init))
            V_unsafe = jnp.ravel(V_state.apply_fn(certificate_params, samples_unsafe))
            V_target = jnp.ravel(V_state.apply_fn(certificate_params, samples_target))
            V_decrease = jnp.ravel(V_state.apply_fn(certificate_params, samples_decrease))

            # Loss in each initial/unsafe state
            if self.exp_certificate:
                losses_init = jnp.maximum(0, V_init + EPS_init)
                losses_unsafe = jnp.maximum(0, - jnp.log(1 - probability_bound) - V_unsafe + EPS_unsafe)
            else:
                losses_init = jnp.maximum(0, V_init - 1 + EPS_init)
                losses_unsafe = jnp.maximum(0, 1 / (1 - probability_bound) - V_unsafe + EPS_unsafe)

            # Loss for expected decrease condition
            expDecr_keys = jax.random.split(noise_key, (self.num_samples_decrease, self.N_expectation))
            actions = Policy_state.apply_fn(policy_params, samples_decrease)
            V_expected = self.loss_exp_decrease_vmap(V_state, certificate_params, samples_decrease, actions,
                                                     expDecr_keys, probability_bound)

            # Compute E[V(x+)] - V(x), approximated over finite number of noise samples
            if self.exp_certificate:
                Vdiffs = jnp.maximum(V_expected - jnp.minimum(V_decrease, jnp.log(3) - jnp.log(1 - probability_bound)) + mesh_loss * (K + lip_certificate) + EPS_decrease, 0)
            else:
                Vdiffs = jnp.maximum(V_expected - jnp.minimum(V_decrease, 3 / (1 - probability_bound)) + mesh_loss * (K + lip_certificate) + EPS_decrease, 0)

            # Determine in which states the expected decrease condition actually applies
            if self.exp_certificate:
                V_decrease_below_thresh = (jax.lax.stop_gradient(V_decrease - mesh_loss * lip_certificate) <= jnp.log(2) - jnp.log(1 - probability_bound))
            else:
                V_decrease_below_thresh = (jax.lax.stop_gradient(V_decrease - mesh_loss * lip_certificate) <= 2 / (1 - probability_bound))

            # Restrict to the expected decrease samples only
            Vdiffs_trim = samples_decrease_bool_not_targetUnsafe * V_decrease_below_thresh * jnp.ravel(Vdiffs)

            #####

            if len(counterexamples) > 0:
                # Certificate values in all counterexample states
                V_cx = jnp.ravel(V_state.apply_fn(certificate_params, cx_samples))

                if self.exp_certificate:
                    V_decrease_cx_below_thresh = (jax.lax.stop_gradient(V_cx - mesh_loss * lip_certificate) <= jnp.log(2) - jnp.log(1 - probability_bound))
                else:
                    V_decrease_cx_below_thresh = (jax.lax.stop_gradient(V_cx - mesh_loss * lip_certificate) <= 2 / (1 - probability_bound))

                # Add initial/unsafe state counterexample loss
                if self.exp_certificate:
                    losses_init_cx = jnp.maximum(0, V_cx + EPS_init)
                    losses_unsafe_cx = jnp.maximum(0, - jnp.log(1 - probability_bound) - V_cx + EPS_unsafe)
                    loss_init = jnp.maximum(jnp.max(losses_init, axis=0), jnp.max(cx_bool_init * losses_init_cx, axis=0))
                    loss_unsafe = -1 / jnp.log(1 - probability_bound) * jnp.maximum(jnp.max(losses_unsafe, axis=0), jnp.max(cx_bool_unsafe * losses_unsafe_cx, axis=0))
                else:
                    losses_init_cx = jnp.maximum(0, V_cx - 1 + EPS_init)
                    losses_unsafe_cx = jnp.maximum(0, 1 / (1 - probability_bound) - V_cx + EPS_unsafe)
                    loss_init = jnp.maximum(jnp.max(losses_init, axis=0), jnp.max(cx_bool_init * losses_init_cx, axis=0))
                    loss_unsafe = (1 - probability_bound) * jnp.maximum(jnp.max(losses_unsafe, axis=0), jnp.max(cx_bool_unsafe * losses_unsafe_cx, axis=0))

                # Add expected decrease loss
                expDecr_keys_cx = jax.random.split(noise_key, (self.batch_size_counterx, self.N_expectation))
                actions_cx = Policy_state.apply_fn(policy_params, cx_samples)
                V_expected = self.loss_exp_decrease_vmap(V_state, certificate_params, cx_samples, actions_cx,
                                                         expDecr_keys_cx, probability_bound)
                if self.exp_certificate:
                    Vdiffs_cx = jnp.maximum(V_expected - jnp.minimum(V_cx, jnp.log(3) - jnp.log(1 - probability_bound)) + mesh_loss * (K + lip_certificate) + EPS_decrease, 0)
                else:
                    Vdiffs_cx = jnp.maximum(V_expected - jnp.minimum(V_cx, 3 / (1 - probability_bound)) + mesh_loss * (K + lip_certificate) + EPS_decrease, 0)
                Vdiffs_cx_trim = cx_bool_decrease * V_decrease_cx_below_thresh * jnp.ravel(Vdiffs_cx)

                if self.loss_decr_squared:
                    loss_exp_decrease_mean = expDecr_multiplier * (
                            jnp.sqrt((jnp.sum(Vdiffs_trim ** 2, axis=0) + jnp.sum(Vdiffs_cx_trim ** 2, axis=0)) \
                                     / (jnp.sum(samples_decrease_bool_not_targetUnsafe * V_decrease_below_thresh, axis=0) + jnp.sum(cx_bool_decrease * V_decrease_cx_below_thresh,
                                                                                                                                    axis=0) + 1e-4) + 1e-4) - 1e-2)
                else:
                    loss_exp_decrease_mean = expDecr_multiplier * (
                            (jnp.sum(Vdiffs_trim, axis=0) + jnp.sum(Vdiffs_cx_trim, axis=0)) \
                            / (jnp.sum(samples_decrease_bool_not_targetUnsafe * V_decrease_below_thresh, axis=0) + jnp.sum(cx_bool_decrease * V_decrease_cx_below_thresh,
                                                                                                                           axis=0) + 1e-4))

                if self.loss_decr_max:
                    loss_exp_decrease_max = jnp.maximum(jnp.max(Vdiffs_trim), jnp.max(Vdiffs_cx_trim))
                else:
                    loss_exp_decrease_max = 0

            else:
                # Add initial/unsafe state counterexample loss
                if self.exp_certificate:
                    f_unsafe = -1 / jnp.log(1 - probability_bound)

                    loss_init = jnp.max(losses_init, axis=0)
                    loss_unsafe = f_unsafe * jnp.max(losses_unsafe, axis=0)
                else:
                    f_unsafe = (1 - probability_bound)

                    loss_init = jnp.max(losses_init, axis=0)
                    loss_unsafe = f_unsafe * jnp.max(losses_unsafe, axis=0)

                if self.loss_decr_squared:
                    loss_exp_decrease_mean = expDecr_multiplier * (
                            jnp.sqrt(jnp.sum(Vdiffs_trim ** 2, axis=0) / (jnp.sum(samples_decrease_bool_not_targetUnsafe * V_decrease_below_thresh, axis=0) + 1e-4) + 1e-4) - 1e-3)
                else:
                    loss_exp_decrease_mean = expDecr_multiplier * (
                            jnp.sum(Vdiffs_trim, axis=0) / (jnp.sum(samples_decrease_bool_not_targetUnsafe * V_decrease_below_thresh, axis=0) + 1e-4))

                if self.loss_decr_max:
                    loss_exp_decrease_max = jnp.max(Vdiffs_trim)
                else:
                    loss_exp_decrease_max = 0

            #####

            # Loss to promote low Lipschitz constant
            loss_lipschitz = self.lambda_lipschitz * (jnp.maximum(lip_certificate - self.max_lip_certificate, 0) +
                                                      jnp.maximum(lip_policy - self.max_lip_policy, 0))

            # Auxiliary losses
            loss_min_target = jnp.maximum(0, jnp.min(V_target, axis=0) - self.glob_min)
            loss_min_init = jnp.maximum(0, jnp.min(V_target, axis=0) - jnp.min(V_init, axis=0))
            loss_min_unsafe = jnp.maximum(0, jnp.min(V_target, axis=0) - jnp.min(V_unsafe, axis=0))
            loss_aux = self.auxiliary_loss * (loss_min_target + loss_min_init + loss_min_unsafe)

            # Define total loss
            loss_total = (loss_init + loss_unsafe + loss_exp_decrease_mean + loss_exp_decrease_max + loss_lipschitz + loss_aux)

            infos = {
                '0. total': loss_total,
                '1. init': loss_init,
                '2. unsafe': loss_unsafe,
                '3. expDecrease_mean': loss_exp_decrease_mean,
                '4. expDecrease_max': loss_exp_decrease_max,
                '5. loss_lipschitz': loss_lipschitz,
            }

            if self.auxiliary_loss > 0:
                infos['8. loss auxiliary'] = loss_aux

            return loss_total, infos

        # Compute gradients
        loss_grad_fun = jax.value_and_grad(loss_fun, argnums=(0, 1), has_aux=True)
        (loss_val, infos), (V_grads, Policy_grads) = loss_grad_fun(V_state.params, Policy_state.params)

        samples_in_batch = {
            'init': samples_init,
            'target': samples_target,
            'unsafe': samples_unsafe,
            'decrease': samples_decrease,
            'decrease_not_in_target': samples_decrease_bool_not_targetUnsafe,
            'counterx': cx_samples,
            'counterx_init': cx_bool_init,
            'counterx_unsafe': cx_bool_unsafe,
            'counterx_decrease': cx_bool_decrease
        }

        return V_grads, Policy_grads, infos, key, samples_in_batch

    def debug_train_step(self, args, samples_in_batch, iteration):
        '''
        Debug function for the training. 

        :param args: Command line arguments given. 
        :param samples_in_batch: Dictionary, giving the samples used in the batch per category. 
        :param iteration: Number of the CEGIS iteration.
        '''

        samples_in_batch['decrease'] = samples_in_batch['decrease'][samples_in_batch['decrease_not_in_target']]

        print('Samples used in last train steps:')
        print(f"- # init samples: {len(samples_in_batch['init'])}")
        print(f"- # unsafe samples: {len(samples_in_batch['unsafe'])}")
        print(f"- # target samples: {len(samples_in_batch['target'])}")
        print(f"- # decrease samples: {len(samples_in_batch['decrease'])}")
        print(f"- # counterexamples: {len(samples_in_batch['counterx'])}")
        print(f"-- # cx init: {len(samples_in_batch['counterx'][samples_in_batch['counterx_init']])}")
        print(f"-- # cx unsafe: {len(samples_in_batch['counterx'][samples_in_batch['counterx_unsafe']])}")
        print(f"-- # cx decrease: {len(samples_in_batch['counterx'][samples_in_batch['counterx_decrease']])}")

        # Plot samples used in batch
        for s in ['init', 'unsafe', 'target', 'decrease', 'counterx']:
            filename = f"plots/{args.start_datetime}_train_debug_{str(s)}_iteration={iteration}"
            plot_dataset(self.env, additional_data=np.array(samples_in_batch[s]), folder=args.cwd, filename=filename)

        for s in ['counterx_init', 'counterx_unsafe', 'counterx_decrease']:
            filename = f"plots/{args.start_datetime}_train_debug_{str(s)}_iteration={iteration}"
            idxs = samples_in_batch[s]
            plot_dataset(self.env, additional_data=np.array(samples_in_batch['counterx'])[idxs], folder=args.cwd,
                         filename=filename)
