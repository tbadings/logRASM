'''
This is the main Python file for running the code.
The file can be run from the terminal as

``Python run.py --model <model-name> ...``

For more details, please see the ReadMe file. For all available arguments, please see the function :func:`core.parse_args.parse_arguments`.
'''

import os
from datetime import datetime

import numpy as np

from core.parse_args import parse_arguments, set_mesh_loss

if __name__ == "__main__":

    # Define argument object
    args = parse_arguments(linfty=False, datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), cwd=os.getcwd())
    if args.high_precision:
        os.environ["JAX_ENABLE_X64"] = "True"
        os.environ['JAX_DEFAULT_DTYPE_BITS'] = '64'

    # Fix CUDNN non-determinism; https://github.com/google/jax/issues/4823#issuecomment-952835771
    os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
    os.environ["TF_CUDNN DETERMINISTIC"] = "1"

    import jax
    import time
    import models
    import sys
    from tqdm import tqdm
    from pathlib import Path

    from core.buffer import Buffer
    from core.commons import args2dict
    from core.jax_utils import orbax_set_config, load_policy_config, create_nn_states
    from core.learner import Learner
    from core.logger import Logger
    from core.plot import plot_certificate_2D, plot_dataset, plot_traces, vector_plot
    from core.ppo_jax import PPO, PPOargs
    from core.verifier import Verifier
    from core.jax_utils import lipschitz_coeff
    from train_SB3 import pretrain_policy
    from validate_certificate import validate_RASM

    # Flax imports
    import flax
    import flax.linen as nn

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import orbax.checkpoint
    from flax.training import orbax_utils
    from jax.extend.backend import get_backend

    # Create output object and set file to export to
    if args.load_ckpt != '':
        experiment_name = f'date={args.start_datetime}_model={args.model}_p={args.probability_bound}_seed={args.seed}_alg=pretrained_policy'
    else:
        experiment_name = f'date={args.start_datetime}_model={args.model}_p={args.probability_bound}_seed={args.seed}_alg={args.pretrain_method}'
    output_folder = Path(args.cwd, 'output', args.logger_prefix, experiment_name)
    args.output_folder = output_folder
    LOGG = Logger(output_folder=output_folder, round_decimals=6)

    envfun = models.get_model_fun(args.model)

    if not args.silent:
        print('\nRun using arguments:')
        for key, val in vars(args).items():
            print(' - `' + str(key) + '`: ' + str(val))
        print(f'\nRunning JAX on device: {get_backend().platform}')
        if args.high_precision:
            test = jnp.array([3.5096874987, 6.30985987], dtype=jnp.float64)
            if test.dtype == jnp.float64:
                print(f'JAX number precision: {test.dtype}')
            else:
                print(f'(!!!) Warning: JAX precision 64-bit requested, but actual precision is: {test.dtype}')
        else:
            print(f'JAX number precision: float32')
        print('\n================\n')

    LOGG.append_time(key='initialize', value=LOGG.get_timer_value())

    # %% ### PPO policy initialization ###

    pi_neurons_per_layer = [args.neurons_per_layer for _ in range(args.hidden_layers)]
    pi_act_funcs_jax = [nn.relu for _ in range(args.hidden_layers)]
    pi_act_funcs_txt = ['relu' for _ in range(args.hidden_layers)]

    if args.load_ckpt != '':
        # Load existing pretrained policy
        checkpoint_path = Path(args.cwd, args.load_ckpt)
        print(f'\n=== READ FROM CHECKPOINT: {checkpoint_path} ===\n')

    elif args.pretrain_method == 'PPO_JAX':
        print(f'Run PPO (JAX) for model `{args.model}`')

        batch_size = int(args.pretrain_num_envs * args.ppo_num_steps_per_batch)
        minibatch_size = int(batch_size // args.ppo_num_minibatches)
        num_iterations = int(args.pretrain_total_steps // batch_size)

        ppo_args = PPOargs(seed=args.seed,
                           high_precision=args.high_precision,
                           layout=args.layout,
                           total_timesteps=args.pretrain_total_steps,
                           learning_rate=3e-4,
                           num_envs=args.pretrain_num_envs,
                           num_steps=args.ppo_num_steps_per_batch,
                           anneal_lr=True,
                           gamma=0.99,
                           gae_lambda=0.95,
                           num_minibatches=args.ppo_num_minibatches,
                           update_epochs=10,
                           clip_coef=0.2,
                           ent_coef=0.0,
                           vf_coef=0.5,
                           max_grad_norm=0.5,
                           weighted=False,  # args.weighted,
                           cplip=False,  # args.cplip,
                           linfty=args.linfty,
                           batch_size=batch_size,
                           minibatch_size=minibatch_size,
                           num_iterations=num_iterations)

        # Only returns the policy state; not the full agent state used in the PPO algorithm.
        _, Policy_state, checkpoint_path = PPO(envfun(args),
                                               args.model,
                                               cwd=args.cwd,
                                               args=ppo_args,
                                               max_policy_lipschitz=args.ppo_max_policy_lipschitz,
                                               neurons_per_layer=pi_neurons_per_layer,
                                               activation_functions_jax=pi_act_funcs_jax,
                                               activation_functions_txt=pi_act_funcs_txt,
                                               verbose=args.ppo_verbose)

        print('\n=== POLICY TRAINING (WITH PPO, JAX) COMPLETED ===\n')
    else:
        print(f'Run {args.pretrain_method} (PyTorch) for model `{args.model}`')

        _, _, _, checkpoint_path = pretrain_policy(
            args,
            env_name=args.model,
            cwd=args.cwd,
            RL_method=args.pretrain_method,
            seed=args.seed,
            num_envs=args.pretrain_num_envs,
            total_steps=args.pretrain_total_steps,
            policy_size=pi_neurons_per_layer,
            activation_fn_jax=pi_act_funcs_jax,
            activation_fn_txt=pi_act_funcs_txt,
            allow_tanh=False)

        print(f'\n=== POLICY TRAINING (WITH {args.pretrain_method}, PYTORCH) COMPLETED ===\n')

    LOGG.append_time(key='pretrain_policy', value=LOGG.get_timer_value())

    # %%

    # Set random seeds
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    cegis_start_time = time.time()

    # Create gym environment (jax/flax version)
    env = envfun(args)

    # Set the mesh_loss
    args = set_mesh_loss(args, env)

    V_neurons_withOut = [args.neurons_per_layer for _ in range(args.hidden_layers)] + [1]
    if args.exp_certificate:
        V_act_fn_withOut = [nn.relu for _ in range(args.hidden_layers)] + [None]
        V_act_fn_withOut_txt = ['relu' for _ in range(args.hidden_layers)] + ['None']
    else:
        V_act_fn_withOut = [nn.relu for _ in range(args.hidden_layers)] + [nn.softplus]
        V_act_fn_withOut_txt = ['relu' for _ in range(args.hidden_layers)] + ['softplus']

    # Load policy configuration and
    Policy_config = load_policy_config(checkpoint_path, key='config')
    V_state, Policy_state, Policy_config, Policy_neurons_withOut = create_nn_states(env, Policy_config, V_neurons_withOut,
                                                                                    V_act_fn_withOut, pi_neurons_per_layer,
                                                                                    Policy_lr=args.Policy_learning_rate,
                                                                                    V_lr=args.V_learning_rate)

    # Restore state of policy network
    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    target = {'model': Policy_state, 'config': Policy_config}
    Policy_state = orbax_checkpointer.restore(checkpoint_path, item=target)['model']

    # Define Learner
    learn = Learner(env, args=args)

    verify = Verifier(env)
    verify.partition_noise(env, args)

    # Define counterexample buffer
    if not args.silent:
        print(f'- Create initial counterexample buffer')
    counterx_buffer = Buffer(dim=env.state_space.dimension,
                             max_size=args.num_counterexamples_in_buffer,
                             extra_dims=4)

    LOGG.export_args(args)  # Export all arguments to CSV file

    # Add standard info of the benchmark
    LOGG.add_info_from_dict({
        'start_date': args.start_datetime,
        'model': args.model,
        'layout': args.layout,
        'seed': args.seed,
        'ckpt': args.load_ckpt,
        'algorithm': args.pretrain_method,
        'probability_bound': args.probability_bound,
        'local_refinement': args.local_refinement,
        'weighted_Lipschitz': args.weighted,
        'cplip': args.cplip,
        'split_lip': args.split_lip,
        'improved_softplus_lip': args.improved_softplus_lip,
        'exp_certificate': args.exp_certificate,
    })

    LOGG.append_time(key='initialize_CEGIS', value=LOGG.get_timer_value())

    # %%

    # Main Learner-Verifier loop
    LOGG.add_info(key='status', value='none')

    for i in range(args.cegis_iterations):
        print(f'\n=== Iter. {i} (time elapsed: {(time.time() - cegis_start_time):.2f} sec.) ===\n')
        if i > 0:
            args.mesh_loss = np.round(np.maximum(args.mesh_loss * args.mesh_loss_decrease_per_iter, args.mesh_loss_min), decimals=8)
        print('- Current mesh for learner loss:', args.mesh_loss)

        # Automatically determine number of batches
        num_batches = int(np.ceil((args.num_samples_per_epoch + len(counterx_buffer.data)) / args.batch_size))

        if not args.silent:
            print(f'- Number of epochs: {args.epochs}; number of batches: {num_batches}')
            print(f'- Auxiliary loss enabled: {args.auxiliary_loss}')
            lip_certificate, _ = lipschitz_coeff(V_state.params, args.weighted, args.cplip, args.linfty)
            lip_policy, _ = lipschitz_coeff(Policy_state.params, args.weighted, args.cplip, args.linfty)
            print(f' - Certificate Lipschitz: {lip_certificate:.2f}')
            print(f' - Policy Lipschitz: {lip_policy:.2f}')

        for j in tqdm(range(args.epochs), desc=f"Learner epochs (iteration {i})"):
            for k in range(num_batches):

                # Main train step function: Defines one loss function for the provided batch of train data and minimizes it
                V_grads, Policy_grads, infos, key, samples_in_batch = learn.train_step(
                    key=key,
                    V_state=V_state,
                    Policy_state=Policy_state,
                    counterexamples=counterx_buffer.data[:, :-1],
                    mesh_loss=args.mesh_loss,
                    probability_bound=args.probability_bound,
                    expDecr_multiplier=args.expDecr_multiplier
                )

                fail = True
                if np.isnan(infos['0. total']):
                    print('(!!!) Warning: The losses contained NaN values, which indicates most probably at an error in the learner module')
                    LOGG.add_info(key='status', value='loss_nan')
                elif np.isinf(infos['0. total']):
                    print('(!!!) Warning: The loss is infinite, which indicates most probably at an error in the learner module')
                    LOGG.add_info(key='status', value='loss_inf')
                elif np.isnan(V_grads['params']['Dense_0']['kernel']).any():
                    print('(!!!) Warning: Gradient contained NaN values')
                    LOGG.add_info(key='status', value='gradient_nan')
                elif np.isinf(V_grads['params']['Dense_0']['kernel']).any():
                    print('(!!!) Warning: Gradient contained infinite values')
                    LOGG.add_info(key='status', value='gradient_inf')
                else:
                    # Update parameters
                    fail = False
                    if args.update_certificate:
                        V_state = V_state.apply_gradients(grads=V_grads)
                    if args.update_policy and i >= args.update_policy_after_iteration:
                        Policy_state = Policy_state.apply_gradients(grads=Policy_grads)

                if fail:
                    # Plot current certificate
                    filename = f"{args.start_datetime}_certificate_iteration={i}"
                    plot_certificate_2D(env, V_state, folder=output_folder, filename=filename, title=(not args.presentation_plots),
                                        labels=(not args.presentation_plots))

                    print(f'\nLoss components in last train step:')
                    for ky, info in infos.items():
                        print(f' - {ky}:', info)

                    print('(!!!) Error: Failure detected, so terminate script')
                    sys.exit()

        if i >= 1 and args.debug_train_step:
            learn.debug_train_step(args, samples_in_batch, iteration=i)

        if not args.silent:
            print(f'Number of times the learn.train_step function was compiled: {learn.train_step._cache_size()}')
            print(f'\nLoss components in last train step:')
            for ky, info in infos.items():
                print(f' - {ky}:', info)  # {info:.8f}')

        LOGG.append_time(key=f'iter{i}_learner', value=LOGG.get_timer_value())
        LOGG.append_Lipschitz(Policy_state, V_state, iteration=i, silent=args.silent)

        # Create plots (only works for 2D model)
        if args.plot_intermediate:
            # Plot traces
            if not args.silent:
                print('- Plot traces...')
            filename = f"{args.start_datetime}_policy_traces_iteration={i}"
            plot_traces(env, Policy_state, key=jax.random.PRNGKey(2), folder=output_folder, filename=filename,
                        title=(not args.presentation_plots))

            # Plot vector plot of policy
            if env.state_dim == 2:
                if not args.silent:
                    print('- Plot vector field...')
                filename = f"{args.start_datetime}_policy_vector_plot_iteration={i}"
                vector_plot(env, Policy_state, folder=output_folder, filename=filename, title=(not args.presentation_plots))

            # Plot base training samples + counterexamples
            if not args.silent:
                print('- Plot counterexamples...')
            filename = f"{args.start_datetime}_counterexample_buffer_iteration={i}"
            plot_dataset(env, additional_data=counterx_buffer.data, folder=output_folder, filename=filename,
                         title=(not args.presentation_plots))

            # Plot current certificate
            if not args.silent:
                print('- Plot certificate...')
            filename = f"{args.start_datetime}_certificate_iteration={i}"
            plot_certificate_2D(env, V_state, folder=output_folder, filename=filename, title=(not args.presentation_plots),
                                labels=(not args.presentation_plots))

        LOGG.append_time(key=f'iter{i}_plot', value=LOGG.get_timer_value())

        finished, counterx, counterx_weights, counterx_hard, total_samples_used, total_samples_naive \
            = verify.check_and_refine(i, env, args, V_state, Policy_state)

        LOGG.append_time(key=f'iter{i}_verifier', value=LOGG.get_timer_value())

        ##########

        # Export final policy and certificate (together in a single checkpoint)
        Policy_config = orbax_set_config(start_datetime=args.start_datetime, env_name=args.model, layout=args.layout,
                                         seed=args.seed, RL_method=args.pretrain_method,
                                         total_steps=args.pretrain_total_steps,
                                         neurons_per_layer=Policy_neurons_withOut,
                                         activation_fn_txt=Policy_config['activation_fn'])

        V_config = orbax_set_config(start_datetime=args.start_datetime, env_name=args.model, layout=args.layout,
                                    seed=args.seed, RL_method=args.pretrain_method,
                                    total_steps=args.pretrain_total_steps,
                                    neurons_per_layer=V_neurons_withOut,
                                    activation_fn_txt=V_act_fn_withOut_txt)

        general_config = args2dict(start_datetime=args.start_datetime, env_name=args.model, layout=args.layout,
                                   seed=args.seed, probability_bound=args.probability_bound,
                                   exp_certificate=args.exp_certificate, weighted=args.weighted, iteration=i)

        ckpt = {'general_config': general_config, 'V_state': V_state, 'Policy_state': Policy_state,
                'V_config': V_config, 'Policy_config': Policy_config}

        final_ckpt_path = Path(output_folder, 'final_ckpt')
        orbax_checkpointer.save(final_ckpt_path, ckpt,
                                save_args=flax.training.orbax_utils.save_args_from_target(ckpt), force=True)
        print(f'- Policy and certificate checkpoint exported to {str(final_ckpt_path)}')

        ##########

        if finished:
            total_time = time.time() - cegis_start_time
            LOGG.add_info(key='status', value='success')
            LOGG.add_info(key='total_CEGIS_time', value=total_time)
            LOGG.add_info(key='verify_samples', value=total_samples_used)
            LOGG.add_info(key='verify_samples_naive', value=total_samples_naive)
            print(f'\nTotal CEGIS (learner-verifier) runtime: {total_time:.2f} sec.')

            if args.validate:
                # Perform at least enough simulations to get a more fine-grained empirical satisfaction probability than needed to show the satisfaction probability
                num_traces = max(1000, int(1 / (1 - args.probability_bound)))
                validate_RASM(final_ckpt_path, num_traces=num_traces, plot_latex_text=args.plot_latex_text)  # Perform validation of RASM

            print('\n=== Successfully learned certificate! ===')

            # Plot final (log)RASM
            if env.state_dim == 2:
                # 2D plot for the certificate function over the state space
                filename = f"{args.start_datetime}_certificate_iteration={i}"
                plot_certificate_2D(env, V_state, folder=output_folder, filename=filename,
                                    title=(not args.presentation_plots),
                                    labels=(not args.presentation_plots))

            break

        else:
            counterx_nonzero_weight = np.sum(counterx_weights, axis=1) > 0
            print(f'- Number of counterexamples with nonzero weight: {np.sum(counterx_nonzero_weight)}')
            if np.sum(counterx_nonzero_weight) > 0:
                counterx = counterx[counterx_nonzero_weight]
                counterx_weights = counterx_weights[counterx_nonzero_weight]
                counterx_hard = counterx_hard[counterx_nonzero_weight]

            # Append weights to the counterexamples
            counterx_plus_weights = np.hstack((counterx[:, :env.state_dim], counterx_weights, counterx_hard.reshape(-1, 1)))

            # Add counterexamples to the counterexample buffer
            if not args.silent:
                print(f'\nRefresh {(args.counterx_refresh_fraction * 100):.1f}% of the counterexample buffer')
            counterx_buffer.append_and_remove(refresh_fraction=args.counterx_refresh_fraction,
                                              samples=counterx_plus_weights,
                                              perturb=args.perturb_counterexamples,
                                              cell_width=counterx[:, -1],
                                              weighted_sampling=args.weighted_counterexample_sampling,
                                              verbose=False)

            if not args.silent:
                print('Counterexample buffer statistics:')
                print(f'- Total counterexamples: {len(counterx_buffer.data)}')
                print(f'- Hard violations: {int(np.sum(counterx_buffer.data[:, -1]))}')
                print(f'- Exp decrease violations: {int(np.sum(counterx_buffer.data[:, -4] > 0))}')
                print(f'- Init state violations: {int(np.sum(counterx_buffer.data[:, -3] > 0))}')
                print(f'- Unsafe state violations: {int(np.sum(counterx_buffer.data[:, -2] > 0))}')

        LOGG.append_time(key=f'iter{i}_process_counterexamples', value=LOGG.get_timer_value())

        plt.close('all')
        if not args.silent:
            print('\n================\n')

    if not finished:
        print(f'\n=== Program not terminated in the maximum number of iterations ({args.cegis_iterations}) ===')

    print('\n============================================================\n')
