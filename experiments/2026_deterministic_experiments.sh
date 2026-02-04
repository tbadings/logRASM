python run.py --seed 1 --model TripleIntegrator --logger_prefix TripleIntegrator --pretrain_total_steps 100000 --hidden_layers 3 --mesh_loss 0.005 --mesh_loss_decrease_per_iter 0.9 --mesh_verify_grid_init 0.04 --noise_partition_cells 6 --max_refine_factor 4 --verify_batch_size 10000 \
    --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --expDecr_multiplier 10 --pretrain_method PPO_JAX --refine_threshold 250000000 --epochs 100 --forward_pass_batch_size 1000000 \
    --probability_bound 0.1 --exp_certificate --deterministic;

python run.py --seed 1 --model Drone4D --layout 2 --logger_prefix Drone4D --pretrain_total_steps 1000000 --hidden_layers 3 --mesh_loss 0.01 --mesh_verify_grid_init 0.06 --refine_threshold 50000000 --verify_threshold 600000000 --noise_partition_cells 12 --max_refine_factor 2 --verify_batch_size 10000 \
    --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --expDecr_multiplier 10 --pretrain_method PPO_JAX --refine_threshold 250000000 --epochs 100 --forward_pass_batch_size 1000000 \
    --probability_bound 0.1 --exp_certificate --deterministic;

python run.py --seed 1 --model PlanarRobot --logger_prefix PlanarRobot --pretrain_total_steps 10000000 --hidden_layers 3 --mesh_loss 0.005 --mesh_loss_decrease_per_iter 0.9 --mesh_verify_grid_init 0.04 --noise_partition_cells 12 --max_refine_factor 4 --verify_batch_size 10000 \
    --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --expDecr_multiplier 10 --pretrain_method PPO_JAX --refine_threshold 250000000 --epochs 100 --forward_pass_batch_size 1000000 \
    --probability_bound 0.1 --exp_certificate --deterministic;