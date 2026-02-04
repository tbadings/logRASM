import argparse
import sys

import numpy as np


def parse_arguments(linfty, datetime, cwd):
    '''
    Function to parse arguments provided.

    :param linfty: If True, use L_infty norms for Lipschitz constants.
    :param datetime: Current datetime.
    :param cwd: Current working directory.
    :return: Arguments object.
    '''

    parser = argparse.ArgumentParser(prefix_chars='--')

    # JAX settings
    parser.add_argument('--high_precision', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, use JAX with 64-bit precision; otherwise, use 32-bit precision")

    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, makes the stochastic noise in the dynamics zero")

    # GENERAL OPTIONS
    parser.add_argument('--model', type=str, default="LinearSystem",
                        help="Gymnasium environment ID")
    parser.add_argument('--layout', type=int, default=0,
                        help="Select a particular layout for the benchmark model (if this option exists)")
    parser.add_argument('--probability_bound', type=float, default=0.9,
                        help="Bound on the reach-avoid probability to verify")
    parser.add_argument('--seed', type=int, default=1,
                        help="Random seed")
    parser.add_argument('--logger_prefix', type=str, default="",
                        help="Prefix to output export file")
    parser.add_argument('--silent', action=argparse.BooleanOptionalAction, default=False,
                        help="Only show crucial output in terminal")
    parser.add_argument('--update_certificate', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, certificate network is updated by the Learner")
    parser.add_argument('--update_policy', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, policy network is updated by the Learner")
    parser.add_argument('--validate', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, automatically perform validation once (log)RASM was successfully learned")

    # PLOT OPTIONS
    parser.add_argument('--plot_intermediate', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plots are generated throughout the CEGIS iterations (increases runtime)")
    parser.add_argument('--presentation_plots', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, labels and titles are omitted from plots (better for generating GIFs for, e.g., presentations)")
    parser.add_argument('--plot_latex_text', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, use latex for labels in plots; If False, use plain text (e.g., Docker container does not have latex installed)")

    ### POLICY INITIALIZATION ARGUMENTS
    parser.add_argument('--load_ckpt', type=str, default='',
                        help="If given, a checkpoint is loaded from this file")
    parser.add_argument('--pretrain_method', type=str, default='PPO_JAX',
                        help="Method to pretrain (initialize) the policy; If different from PPO_JAX, it tries to use StableBaselines3")
    parser.add_argument('--pretrain_total_steps', type=int, default=1_000_000,
                        help="Total number of steps for pretraining the policy")
    parser.add_argument('--pretrain_num_envs', type=int, default=10,
                        help="Number of parallel environments in PPO (for policy initialization")

    ### JAX PPO arguments
    parser.add_argument('--ppo_max_policy_lipschitz', type=float, default=10,
                        help="Max. Lipschitz constant for policy to train towards in PPO (below this value, loss is zero)")
    parser.add_argument('--ppo_num_steps_per_batch', type=int, default=2048,
                        help="Total steps for rollout in PPO (for policy initialization")
    parser.add_argument('--ppo_num_minibatches', type=int, default=32,
                        help="Number of minibatches in PPO (for policy initialization")
    parser.add_argument('--ppo_verbose', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, print more output during PPO (JAX) training")

    ### VERIFIER MESH SIZES
    parser.add_argument('--mesh_loss', type=float, default=0,
                        help="Specifying the mesh size used in the loss function")
    parser.add_argument('--mesh_loss_decrease_per_iter', type=float, default=1,
                        help="Specifying the decrease factor in the mesh size used in the loss function per iteration (one means no decrease)")
    parser.add_argument('--mesh_loss_min', type=float, default=0.00001,
                        help="Specifying the minimum mesh size used in the loss function")
    parser.add_argument('--tauK_loss', type=float, default=0,
                        help="Specifying the value of mesh/tau*K used in the loss function")
    parser.add_argument('--tauK_policy_lipschitz', type=float, default=0,
                        help="The assumed minimal policy Lipschitz constant for computing the mesh_loss (only used when tauK_loss is specified)")
    parser.add_argument('--mesh_verify_grid_init', type=float, default=0.01,
                        help="Initial mesh size for verifying grid. Mesh is defined such that |x-y|_1 <= tau for any x in X and discretized point y")

    ### REFINE ARGUMENTS
    parser.add_argument('--not_refine_before_iter', type=int, default=0,
                        help="Do not perform refinements in the verifier before the specified iteration")
    parser.add_argument('--mesh_refine_min', type=float, default=1e-9,
                        help="Lowest allowed verification grid mesh size in the final verification")
    parser.add_argument('--max_refine_factor', type=float, default=float('nan'),
                        help="Maximum value to split each grid point into (per dimension), during the (local) refinement")

    ### LEARNER ARGUMENTS
    parser.add_argument('--cegis_iterations', type=int, default=1000,
                        help="Number of CEGIS iteration to run")
    parser.add_argument('--epochs', type=int, default=25,
                        help="Number of epochs to run in each iteration")
    parser.add_argument('--num_samples_per_epoch', type=int, default=90000,
                        help="Total number of samples to train over in each epoch")
    parser.add_argument('--num_counterexamples_in_buffer', type=int, default=30000,
                        help="Number of counterexamples to keep in the buffer")
    parser.add_argument('--batch_size', type=int, default=4096,
                        help="Batch size used by the learner in each epoch")
    parser.add_argument('--Policy_learning_rate', type=float, default=5e-5,
                        help="Learning rate for changing the policy in the CEGIS loop")
    parser.add_argument('--V_learning_rate', type=float, default=5e-4,
                        help="Learning rate for changing the certificate in the CEGIS loop")
    parser.add_argument('--loss_lipschitz_lambda', type=float, default=0,
                        help="Factor to multiply the Lipschitz loss component with")
    parser.add_argument('--loss_lipschitz_certificate', type=float, default=15,
                        help="When the certificate Lipschitz coefficient is below this value, then the loss is zero")
    parser.add_argument('--loss_lipschitz_policy', type=float, default=4,
                        help="When the policy Lipschitz coefficient is below this value, then the loss is zero")
    parser.add_argument('--expDecr_multiplier', type=float, default=1,
                        help="Multiply the weighted counterexample expected decrease loss by this value")
    parser.add_argument('--debug_train_step', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, generate additional plots for the samples used in the last train step of an iteration")
    parser.add_argument('--auxiliary_loss', type=int, default=0,
                        help="If nonzero, auxiliary loss is added to the learner loss function")
    parser.add_argument('--loss_decr_squared', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, use squared mean in exp. decrease loss; otherwise, use normal mean")
    parser.add_argument('--loss_decr_max', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, also penalize the maximum (instead of the mean) exp. decrease violation")
    parser.add_argument('--eps_decrease', type=float, default=0,
                        help="Epsilon to the expected decrease loss function")
    parser.add_argument('--learner_N_expectation', type=int, default=16,
                        help="Number of samples to use in expectation computation in learner")

    ### VERIFIER ARGUMENTS
    parser.add_argument('--verify_batch_size', type=int, default=30000,
                        help="Number of states for which the verifier checks exp. decrease condition in the same batch")
    parser.add_argument('--forward_pass_batch_size', type=int, default=1_000_000,
                        help="Batch size for performing forward passes on the neural network (reduce if this gives memory issues)")
    parser.add_argument('--noise_partition_cells', type=int, default=12,
                        help="Number of cells to partition the noise space in per dimension (to numerically integrate stochastic noise)")
    parser.add_argument('--counterx_refresh_fraction', type=float, default=0.50,
                        help="Fraction of the counter example buffer to renew after each iteration")
    parser.add_argument('--counterx_fraction', type=float, default=0.25,
                        help="Fraction of counter examples, compared to the total train data set")
    parser.add_argument('--refine_threshold', type=float, default=1e9,
                        help="Do not refine if the number of counterexamples is above this threshold")
    parser.add_argument('--verify_threshold', type=float, default=1e9,
                        help="Do not verify if the number of points to check on is above this threshold")

    ### LEARNER-VERIFIER CASES
    parser.add_argument('--exp_certificate', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, train a logRASM (i.e., exponential certificate); If False, use a standard RASM")
    parser.add_argument('--weighted', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, use weighted norms to compute Lipschitz constants")
    parser.add_argument('--cplip', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, use CPLip method to compute Lipschitz constants")

    ### ARGUMENTS TO EXPERIMENT WITH ###
    parser.add_argument('--local_refinement', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, local grid refinements are performed")
    parser.add_argument('--perturb_counterexamples', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, counterexamples are perturbed before being added to the counterexample buffer")
    parser.add_argument('--min_lip_policy_loss', type=float, default=0,
                        help="Minimum Lipschitz constant policy used in loss function learner")
    parser.add_argument('--hard_violation_multiplier', type=float, default=10,
                        help="Factor to multiply the counterexample weights for hard violations with")
    parser.add_argument('--weighted_counterexample_sampling', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, use weighted sampling of counterexamples")
    parser.add_argument('--min_fraction_samples_per_region', type=float, default=0,
                        help="Minimum fraction of samples in learner for each region/condition")

    ### NEURAL NETWORK ARCHITECTURE
    parser.add_argument('--neurons_per_layer', type=int, default=128,
                        help="Number of neurons per (hidden) layer")
    parser.add_argument('--hidden_layers', type=int, default=2,
                        help="Number of hidden layers")

    ## LIPSCHITZ COEFFICIENT ARGUMENTS
    parser.add_argument('--split_lip', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, use L_f split over the system state space and control action space")
    parser.add_argument('--improved_softplus_lip', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, use improved (local) Lipschitz constants for softplus in V (if False, "
                             "global constant of 1 is used); Can only be used without logRASMs enabled")

    args = parser.parse_args()

    args.linfty = linfty  # Use L1 norm for Lipschitz constants (there is unused but experimental support for L_infty norms)
    args.start_datetime = datetime
    args.cwd = cwd

    # Set refinement factor depending on whether local refinement is enabled
    if args.local_refinement and args.max_refine_factor != args.max_refine_factor:  # a != a means that a is NaN
        args.max_refine_factor = 10
    elif not args.local_refinement and args.max_refine_factor != args.max_refine_factor:  # a != a means that a is NaN
        args.max_refine_factor = np.sqrt(2)

    if args.mesh_loss != 0 and args.tauK_loss != 0:
        print('Error: Cannot specify both mesh_loss and tauK_loss (remove one of the arguments and try again).')
        sys.exit()
    elif args.mesh_loss == 0 and args.tauK_loss == 0:
        print('Error: mesh_loss and tauK_loss are both unspecified (specify one of them and try again).')
        sys.exit()

    if args.mesh_loss_decrease_per_iter == 1 and args.mesh_loss < args.mesh_loss_min:
        print(f'- mesh_loss_min was larger than mesh_loss, so set mesh_loss_min to {args.mesh_loss}')
        args.mesh_loss_min = args.mesh_loss

    if args.exp_certificate and args.improved_softplus_lip:
        args.improved_softplus_lip = False
        print("Warning: No softplus used when using exponential certificate, turned off softplus Lipschitz improvement.")

    if args.deterministic:
        print(f'- Run in deterministic mode (no stochastic noise in dynamics; avoid discretizing the noise space).')
        args.noise_partition_cells = 1
        args.learner_N_expectation = 1

    return args


def set_mesh_loss(args, env, recommended_tauK=0.1):
    """
    Set the mesh / tau for the learner loss function, either based on the specified tau*K value or the value of tau directly

    :param args: Command line arguments given. 
    :param env: Environment.
    """

    MIN_MESH = 0.00001
    MAX_MESH = 0.01

    print('\nEvaluate settings for the mesh/tau used in the learner loss')
    print(f'- Model: {args.model}; layout: {args.layout}')
    print(f'- Probability bound: {args.probability_bound}')

    # Determine level of unsafe states
    if args.exp_certificate:
        v_unsafe = - np.log(1 - args.probability_bound)
        v_init = 0
    else:
        v_unsafe = 1 / (1 - args.probability_bound)
        v_init = 1
    print(f'- Minimum value of certificate in unsafe states: {v_unsafe:.3f}')
    print(f'- Maximum value of certificate in initial states: {v_init:.3f}')

    # Compute minimum value of K needed for this probability bound
    if args.split_lip:
        min_K = (v_unsafe - v_init) / env.init_unsafe_dist * (
                env.lipschitz_f_l1_A + env.lipschitz_f_l1_B * args.tauK_policy_lipschitz)
    else:
        min_K = (v_unsafe - v_init) / env.init_unsafe_dist * (
                env.lipschitz_f_l1 * (1 + args.tauK_policy_lipschitz))
    print(f'- Minimum value of K required is {min_K:.3f}')

    if args.tauK_loss != 0:
        # Set the value of mesh_loss/tau such that tau*min_K equals the desired value
        args.mesh_loss = args.tauK_loss / min_K
        print(f'- Desired value of tau*K is {args.tauK_loss:.5f}')
        print(f'- Calculated mesh size is {args.mesh_loss:.7f}')
        if args.mesh_loss < MIN_MESH:
            args.mesh_loss = MIN_MESH
            print(f'-- Below that min. of {MIN_MESH}, so cap the mesh at this value')
        if args.mesh_loss > MAX_MESH:
            args.mesh_loss = MAX_MESH
            print(f'-- Below that min. of {MAX_MESH}, so cap the mesh at this value')

    else:

        print(f'- Current value of mesh/tau is {(args.mesh_loss):.7f}')
        print(f'- Suggested maximum mesh/tau is {(recommended_tauK / min_K):.7f} (with max. tau*K of {recommended_tauK:.3f})')

        # Use the specified value of mesh_loss/tau directly, in which case we only give hints about whether the value
        # is too high or not.
        if args.mesh_loss * min_K > 1:
            print(f'- Severe warning: The value of mesh_loss ({args.mesh_loss}) is much too high, which likely makes it very hard to learn a proper certificate.')
        elif args.mesh_loss * min_K > 0.2:
            print(f'- Warning: The value of mesh_loss ({args.mesh_loss}) is likely too high for good convergence of the loss to zero.')

    print('')

    return args
