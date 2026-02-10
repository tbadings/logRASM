# To generate figures for in paper, run something like:
# python validate_certificate.py --check_folder '../results/Ablation (Jan 2)/main/' --no-validate_conditions --plot_RASM

import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ['JAX_DEFAULT_DTYPE_BITS'] = '64'

import jax
import jax.numpy as jnp
import argparse
from pathlib import Path
import orbax.checkpoint
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from core.jax_utils import orbax_parse_activation_fn, load_policy_config, create_nn_states
from core.buffer import define_grid_jax
from core.verifier import batched_forward_pass
from core.plot import plot_boxes, plot_certificate_2D, plot_layout, plot_heatmap
from core.commons import Namespace

# Import all benchmark models
import models


class Simulator:
    '''
    Monte Carlo simulator for the reach-avoid specification.
    '''

    def __init__(self, env, Policy_state, key, init_space, xInits=10000):

        self.key, subkey = jax.random.split(key)
        self.env = env
        self.Policy_state = Policy_state
        self.stats = {
            'total_sims': 0,
            'satisfied': 0
        }

        try:
            N_per_set = np.ceil(xInits / len(init_space.sets)).astype(int)
            M = tuple(N_per_set for _ in range(len(init_space.sets)))
            self.initial_states = init_space.sample(subkey, N=M)
        except:
            self.initial_states = init_space.sample(subkey, N=xInits)

    def sample_xInit(self):
        idx = np.random.randint(len(self.initial_states))
        return self.initial_states[idx]

    def empirical_reachability(self, num_traces, horizon):

        traces = {}
        actions = {}
        total_sat = 0

        for n in tqdm(range(num_traces)):
            traces[n], actions[n], sat = self.simulate_trace(horizon)
            total_sat += sat

        empirical_sat = total_sat / num_traces

        return traces, actions, empirical_sat

    def simulate_trace(self, horizon):

        x = np.zeros((horizon + 1, self.env.state_dim))
        a = np.zeros((horizon, self.env.action_space.shape[0]))
        satisfied = False
        done = False

        # Reset environment by sampling an initial state
        x[0] = self.sample_xInit()

        # Check if initial state is already unsafe or in target
        if not self.is_safe(x[0]):
            done = True
        if self.is_goal(x[0]):
            done = True
            satisfied = True

        t = 0
        while not done:
            t += 1

            # Get state and action
            a[t - 1] = self.Policy_state.apply_fn(self.Policy_state.params, x[t - 1])

            # Make step in environment
            x[t], self.key = self.env.step_noise_key(x[t - 1], self.key, a[t - 1])

            # Check if initial state is already unsafe or in target
            if not self.is_safe(x[t]):
                done = True
            if self.is_goal(x[t]):
                done = True
                satisfied = True
            elif t >= horizon:
                done = True

        # Trim trace
        x = x[:t + 1]
        a = a[:t]

        return x, a, satisfied

    def is_safe(self, x):
        ''' Check if given state x is still safe '''
        safe = not self.check_unsafe(x) * self.check_in_state_space(x)
        return safe

    def is_goal(self, x):
        return self.env.target_space.contains(np.array([x]), return_indices=True)[0]

    def check_unsafe(self, x):
        return self.env.unsafe_space.contains(np.array([x]), return_indices=True)[0]

    def check_in_state_space(self, x):
        return self.env.state_space.contains(np.array([x]), return_indices=True)[0]

    def plot(self, traces, num_to_plot, folder, filename):

        dim = self.env.plot_dim

        if dim != 3:
            ax = plt.figure().add_subplot()

            for n in range(num_to_plot):
                plt.plot(traces[n][:, 0], traces[n][:, 1], color="gray", linewidth=1, markersize=1)

            # Plot relevant state sets
            plot_boxes(self.env, ax)

            # Goal x-y limits
            low = self.env.state_space.low
            high = self.env.state_space.high
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[1], high[1])

            ax.set_title(f"Simulated traces ({filename})", fontsize=10)
            if hasattr(self.env, 'variable_names'):
                plt.xlabel(self.env.variable_names[0])
                plt.ylabel(self.env.variable_names[1])

        else:
            ax = plt.figure().add_subplot(projection='3d')

            for n in range(num_to_plot):
                plt.plot(traces[n][:, 0], traces[n][:, 1], traces[n][:, 2], color="gray", linewidth=1, markersize=1)

            # Goal x-y limits
            low = self.env.state_space.low
            high = self.env.state_space.high
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[1], high[1])
            ax.set_zlim(low[2], high[2])

            ax.set_title(f"Simulated traces ({filename})", fontsize=10)
            if hasattr(self.env, 'variable_names'):
                ax.set_xlabel(self.env.variable_names[0])
                ax.set_ylabel(self.env.variable_names[1])
                ax.set_zlabel(self.env.variable_names[2])

        # Export figure
        if folder and filename:
            # Save figure
            for form in ['pdf', 'png']:
                filepath = Path(folder, filename)
                filepath = filepath.parent / (filepath.name + '.' + form)
                plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)


def loss_exp_decrease(env, exp_certificate, V_state, V_params, x, u, noise_key, probability_bound):
    '''
    Compute expected decrease condition

    :param env: Environment.
    :param exp_certificate: True if logRASM is used.
    :param V_state: Certificate neural network.
    :param V_params: Certificate neural network parameters.
    :param x: State.
    :param u: Control input.
    :param noise_key: Key of the random number generator.
    :param probability_bound: The probability bound of the specification that we aim to certify.
    :return: Expected decrease
    '''

    # For each given noise_key, compute the successor state for the pair (x,u)
    state_new, noise_key = env.vstep_noise_batch(x, noise_key, u)

    # Function apply_fn does a forward pass in the certificate network for all successor states in state_new,
    # which approximates the value of the certificate for the successor state (using different noise values).
    # Then, the loss term is zero if the expected decrease in certificate value is at least tau*K.
    if exp_certificate:
        V_expected = jnp.log(jnp.mean(jnp.exp(jnp.minimum(V_state.apply_fn(V_params, state_new), -jnp.log(1 - probability_bound)))))
    else:
        V_expected = jnp.mean(jnp.minimum(V_state.apply_fn(V_params, state_new), 1 / (1 - probability_bound)))

    return V_expected


loss_exp_decrease_vmap = jax.jit(jax.vmap(loss_exp_decrease, in_axes=(None, None, None, None, 0, 0, 0, None), out_axes=0))


def validate_RASM(checkpoint_path, cell_width=0.01, batch_size=10000, forward_pass_batch_size=1_000_000,
                  expected_decrease_samples=100, seed=1, num_traces=1000, plot_RASM=False, plot_latex_text=True, validate_conditions=True):
    '''
    Validate the given (log)RASM by discretizing the state space and computing the expected decrease condition empirically.

    :param checkpoint_path: Path to load checkpoint from.
    :param cell_width: Discretization cell width.
    :param batch_size: Batch size to use for computing the expected decrease condition.
    :param forward_pass_batch_size: Batch size to use for forward passes.
    :param expected_decrease_samples: Number of samples to use for computing the expected decrease condition in each grid point.
    :param seed: Random seed.
    :param num_traces: Number of Monte Carlo simulations to run.
    :param plot_RASM: If True, plot the (log)RASM.
    :param plot_latex_text: If True, use LaTeX for plotting.
    :param validate_conditions: If True, validate the certificate conditions empirically.
    '''

    print(f'- Validate checkpoint in folder "{checkpoint_path}"')

    test = jnp.array([3.5096874987, 6.30985987], dtype=jnp.float64)
    print(f'- Validation uses JAX data type: {test.dtype}\n')

    key = jax.random.PRNGKey(seed)

    LOGG = {}

    Policy_config = load_policy_config(checkpoint_path, key='Policy_config')
    V_config = load_policy_config(checkpoint_path, key='V_config')
    general_config = load_policy_config(checkpoint_path, key='general_config')

    # Create gym environment (jax/flax version)
    envfun = models.get_model_fun(Policy_config['env_name'])

    # Define empty namespace and store layout attribute
    args = Namespace
    args.layout = general_config['layout']

    # Build environment
    env = envfun(args)

    V_neurons_withOut = V_config['neurons_per_layer']
    V_act_fn_withOut_txt = V_config['activation_fn']
    V_act_fn_withOut = orbax_parse_activation_fn(V_act_fn_withOut_txt)

    pi_neurons_withOut = Policy_config['neurons_per_layer']
    pi_act_funcs_txt = Policy_config['activation_fn']
    pi_act_funcs_jax = orbax_parse_activation_fn(pi_act_funcs_txt)

    # Load policy configuration
    V_state, Policy_state, Policy_config, Policy_neurons_withOut = create_nn_states(env, Policy_config,
                                                                                    V_neurons_withOut,
                                                                                    V_act_fn_withOut,
                                                                                    pi_neurons_withOut)

    # Restore state of policy and certificate network
    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    target = {'general_config': general_config, 'V_state': V_state, 'Policy_state': Policy_state, 'V_config': V_config,
              'Policy_config': Policy_config}

    Policy_state = orbax_checkpointer.restore(checkpoint_path, item=target)['Policy_state']
    V_state = orbax_checkpointer.restore(checkpoint_path, item=target)['V_state']

    # %%

    print('Start validation...')
    start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = checkpoint_path.parents[0]

    if plot_RASM:
        figure_folder = 'output/'

        args_file = Path(folder, 'args.csv')

        try:
            args = pd.read_csv(args_file, index_col=0)['arguments']
            weighted = args['weighted']
            exp = args['exp_certificate']
        except:
            weighted = 'unknown'
            exp = 'unknown'

        # Create plot of reach-avoid specification
        title = f"{general_config['env_name']}"
        filename = f"layout_{general_config['env_name']}-{general_config['layout']}_p={str(general_config['probability_bound'])}_weighted={weighted}_exp={exp}_seed={general_config['seed']}"
        plot_layout(env, folder=figure_folder, filename=filename, title=title, latex=plot_latex_text, size=28)

        # Create plot of RASM
        title = f"{general_config['env_name']} ({general_config['probability_bound'] * 100}\%)"
        filename = f"RASM_{general_config['env_name']}-{general_config['layout']}_p={general_config['probability_bound']}_weighted={weighted}_exp={exp}_seed={general_config['seed']}"
        print(filename)
        plot_certificate_2D(env, V_state, folder=figure_folder, filename=filename, logscale=False, title=title, latex=plot_latex_text, size=28, contour=True)

    if validate_conditions:

        # Check conditions for many sampled points
        num_per_dimension = np.array(
            np.ceil((env.state_space.high - env.state_space.low) / cell_width), dtype=int)
        grid = define_grid_jax(env.state_space.low + 0.5 * cell_width, env.state_space.high - 0.5 * cell_width,
                               size=num_per_dimension)
        print(f'- Validation grid points defined ({len(grid)} points)')

        print(f"- Validate certificate for probability bound p={general_config['probability_bound']}")
        if general_config['exp_certificate']:
            print('-- Assuming that we learned log(V)')
        else:
            print('-- Assuming that we learned V directly')
        LOGG['probability bound'] = general_config['probability_bound']

        # Determine at which points to check which condition
        check_decrease = env.target_space.not_contains(grid)
        check_init = env.init_space.contains(grid)
        check_unsafe = env.unsafe_space.contains(grid)
        print('- Grid points split between decrease/init/unsafe states')

        # Check initial state conditions
        V_init = batched_forward_pass(V_state.apply_fn, V_state.params, check_init, 1, forward_pass_batch_size).flatten()
        if general_config['exp_certificate']:
            V_init_violations = int(np.sum(V_init > 0))
        else:
            V_init_violations = int(np.sum(V_init > 1))
        print(f'- {V_init_violations} initial state violations')
        LOGG['init violations'] = V_init_violations

        # Check unsafe state conditions
        V_unsafe = batched_forward_pass(V_state.apply_fn, V_state.params, check_unsafe, 1,
                                        forward_pass_batch_size).flatten()
        if general_config['exp_certificate']:
            V_unsafe_violations = int(np.sum(V_unsafe < -np.log(1 - general_config['probability_bound'])))
        else:
            V_unsafe_violations = int(np.sum(V_unsafe < 1 / (1 - general_config['probability_bound'])))
        print(f'- {V_unsafe_violations} unsafe state violations')
        LOGG['unsafe violations'] = V_unsafe_violations

        # Check expected decrease conditions
        V_expDecr = batched_forward_pass(V_state.apply_fn, V_state.params, check_decrease, 1,
                                         forward_pass_batch_size).flatten()
        actions = batched_forward_pass(Policy_state.apply_fn, Policy_state.params, check_decrease,
                                       env.action_space.shape[0], forward_pass_batch_size)

        # Create batches
        V_expDecrNext = np.zeros(len(check_decrease))
        num_batches = np.ceil(len(check_decrease) / batch_size).astype(int)
        starts = np.arange(num_batches) * batch_size
        ends = np.minimum(starts + batch_size, len(check_decrease))
        for (i, j) in tqdm(zip(starts, ends), total=len(starts), desc='Compute E[V(x_{k+1})]'):
            x = check_decrease[i:j]
            u = actions[i:j]

            key, subkey = jax.random.split(key)
            expDecr_keys = jax.random.split(subkey, (len(x), expected_decrease_samples))

            V_expDecrNext[i:j] = loss_exp_decrease_vmap(env, general_config['exp_certificate'], V_state, V_state.params, x,
                                                        u, expDecr_keys, general_config['probability_bound'])

        # Check which points are also below the unsafe state region threshold
        if general_config['exp_certificate']:
            below_unsafe_threshold = (V_expDecr < - np.log(1 - general_config['probability_bound']))
        else:
            below_unsafe_threshold = (V_expDecr < 1 / (1 - general_config['probability_bound']))

        print('- Points where V(x) is below the unsafe state threshold:', np.sum(below_unsafe_threshold))

        V_expDecr_violations = int(np.sum((V_expDecrNext > V_expDecr) * below_unsafe_threshold))
        print(f'- {V_expDecr_violations} expected decrease violations')
        LOGG['exp. decr. violations'] = V_expDecr_violations

        # Create heatmaps for expected decrease points
        if actions.shape[1] == 1:
            filename = f"actions_iteration={general_config['iteration']}"
            actions_clip = np.clip(actions.flatten(), env.action_space.low, env.action_space.high)
            plot_heatmap(env, check_decrease, actions_clip, folder=folder, filename=filename, title=True, size=20)
        filename = f"Vdiff_iteration={general_config['iteration']}"
        plot_heatmap(env, check_decrease, V_expDecrNext.flatten() - V_expDecr, folder=folder, filename=filename, title=True,
                     size=20)

        if V_init_violations == 0 and V_unsafe_violations == 0 and V_expDecr_violations == 0:
            print('\nCertificate successfully validated!')
            LOGG['correct certificate'] = True
        else:
            LOGG['correct certificate'] = False
            print('\nCertificate is incorrect (conditions are violated; see above)')

        # %%

        if num_traces > 0:
            sim = Simulator(env, Policy_state, key, init_space=env.init_space, xInits=10000)
            traces, actions, empirical_sat = sim.empirical_reachability(num_traces=num_traces, horizon=100)
            print(f'- Empirical satisfaction probability: {empirical_sat:.3f} (out of {num_traces} simulations)')
            LOGG['empirical satisfaction'] = empirical_sat

            filename = f"validation_traces_{start_datetime}"
            sim.plot(traces, num_to_plot=10, folder=folder, filename=filename)

        # Export log to csv
        LOGG_series = pd.Series(LOGG, name="validate log")
        LOGG_series.to_csv(Path(folder, 'validation_log.csv'))

    return


# %%
def validate_all(args, final_ckpt_name):
    '''
    Validate all learned certificates stored in the output folder (which contain the given prefix)
    '''

    check_folder = Path(args.cwd, args.check_folder)
    subfolders = [f.path for f in os.scandir(check_folder) if f.is_dir()]
    validate_ckpts = []

    # Iterate over all folders
    for folder in subfolders:
        # Extract lowest-level folder
        relfolder = Path(folder).name

        # Check if folder contains given prefix (or if no prefix is required)
        if len(args.prefix) == 0 or relfolder.startswith(args.prefix):
            subsubfolders = [Path(f.path).name for f in os.scandir(folder) if f.is_dir()]

            # If final checkpoint is in the subfolder, add it to the list
            if final_ckpt_name in subsubfolders:
                validate_ckpts += [Path(folder, final_ckpt_name)]

    print(f'Found {len(validate_ckpts)} checkpoint(s) to validate')

    for i, checkpoint_path in enumerate(validate_ckpts):
        print(f'\nCheckpoint #{i}: {checkpoint_path}')
        print('==========================')
        validate_RASM(checkpoint_path=checkpoint_path, cell_width=args.cell_width, batch_size=args.batch_size,
                      expected_decrease_samples=args.expected_decrease_samples, seed=args.seed,
                      num_traces=args.num_simulations, validate_conditions=args.validate_conditions, plot_RASM=args.plot_RASM, plot_latex_text=args.plot_latex_text)


# %%

if __name__ == "__main__":
    '''
    Empirically validate learned (log)RASMs for all checkpoints in the given folder, or for a single specified checkpoint.
    '''

    parser = argparse.ArgumentParser(prefix_chars='--')

    # First two arguments are for validating *all checkpoints* in the provided folder
    parser.add_argument('--check_folder', type=str, default='',
                        help="Folder to load orbax checkpoints from.")
    parser.add_argument('--prefix', type=str, default='',
                        help="Only check checkpoints whose output folder starts with the given prefix.")
    #
    parser.add_argument('--checkpoint', type=str, default='',
                        help="File to load orbax checkpoint from")
    parser.add_argument('--cell_width', type=float, default=0.01,
                        help="Cell width of partitioning (logRASM/RASM conditions are checked for every cell center)")
    parser.add_argument('--num_simulations', type=int, default=10000,
                        help="Number of traces to simulate (for empirically estimating the reach-avoid probability).")
    parser.add_argument('--batch_size', type=int, default=10000,
                        help="Number of states for which to check exp. decrease condition in the same batch.")
    parser.add_argument('--forward_pass_batch_size', type=int, default=1000000,
                        help="Batch size for forward passes in neural network.")
    parser.add_argument('--expected_decrease_samples', type=int, default=100,
                        help="Number of samples to validate expected decrease condition with.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Random seed")

    parser.add_argument('--plot_RASM', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot RASM.")
    parser.add_argument('--validate_conditions', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, RASM conditions are validated.")
    parser.add_argument('--plot_latex_text', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, use LaTeX for plotting.")

    args = parser.parse_args()
    args.cwd = os.getcwd()

    # Check whether to check a single checkpoint or a complete folder
    if len(args.check_folder) != 0:

        check_folder = Path(args.cwd, args.check_folder)

        print(f'- Validate all checkpoints in folder "{check_folder}"')
        if len(args.prefix) > 0:
            print(f'- Restrict to folders with prefix: "{args.prefix}"')

        if len(args.checkpoint) != 0:
            print("Warning: --checkpoint argument ignored when also specifying --check_folder.")

        # Check complete folder
        validate_all(args, final_ckpt_name='final_ckpt')

    else:

        # Check single checkpoint
        checkpoint_path = Path(args.cwd, args.checkpoint)

        validate_RASM(checkpoint_path=checkpoint_path, cell_width=args.cell_width, batch_size=args.batch_size,
                      expected_decrease_samples=args.expected_decrease_samples, seed=args.seed,
                      num_traces=args.num_simulations, validate_conditions=args.validate_conditions, plot_RASM=args.plot_RASM, plot_latex_text=args.plot_latex_text)
