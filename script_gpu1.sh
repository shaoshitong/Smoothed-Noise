# CUDA_VISIBLE_DEVICES=1 python animatediff_xl_noise_optimization_150.py --tag "g7_dpm_solver" \
#     --recall_timesteps 1 --ensemble 10 --momentum 0.05  --traj_momentum 0.01 --ensemble_rate 0.05 --fast_ensemble

# CUDA_VISIBLE_DEVICES=1 python animatediff_xl_noise_optimization_150.py --tag "g7_dpm_solver" \
#     --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.05 --fast_ensemble

# CUDA_VISIBLE_DEVICES=1 python animatediff_xl_noise_optimization_150.py --tag "g7_dpm_solver" \
#     --recall_timesteps 1 --ensemble 30 --momentum 0.05  --traj_momentum 0.1 --ensemble_rate 0.05 --fast_ensemble

# CUDA_VISIBLE_DEVICES=1 python animatediff_xl_noise_optimization_150.py --tag "g7_dpm_solver" \
#     --recall_timesteps 1 --ensemble 10 --momentum 0.25  --traj_momentum 0.1 --ensemble_rate 0.05 --fast_ensemble

# CUDA_VISIBLE_DEVICES=1 python animatediff_xl_noise_optimization_150.py --tag "g7_dpm_solver" \
#     --recall_timesteps 1 --ensemble 10 --momentum 0.05  --traj_momentum 0.1 --ensemble_rate 0.05 --fast_ensemble

# CUDA_VISIBLE_DEVICES=1 python animatediff_xl_noise_optimization_150.py --tag "g7_dpm_solver" \
#     --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.1 --ensemble_rate 0.05 --fast_ensemble

CUDA_VISIBLE_DEVICES=1 python animatediff_xl_noise_optimization_ucf.py --tag "ucf_motivation" \
    --recall_timesteps 1 --ensemble 1 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025

CUDA_VISIBLE_DEVICES=2 python animatediff_xl_noise_optimization_ucf2.py --tag "ucf_motivation" \
    --recall_timesteps 1 --ensemble 1 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025