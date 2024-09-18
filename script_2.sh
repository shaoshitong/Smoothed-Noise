CUDA_VISIBLE_DEVICES=6 python latte_noise_optimization_150.py --tag "latte22_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05  --traj_momentum 0.05 --ensemble_rate 0.05 --fast_ensemble &
CUDA_VISIBLE_DEVICES=7  python latte_noise_optimization_150.py --tag "latte22_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.05  --traj_momentum 0.05 --ensemble_rate 0.1 --fast_ensemble &
CUDA_VISIBLE_DEVICES=4 python latte_noise_optimization_150.py --tag "latte22_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05  --traj_momentum 0.95 --ensemble_rate 0.05 --fast_ensemble &
CUDA_VISIBLE_DEVICES=5  python latte_noise_optimization_150.py --tag "latte22_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.05  --traj_momentum 0.95 --ensemble_rate 0.1 --fast_ensemble


CUDA_VISIBLE_DEVICES=6 python latte_noise_optimization_150.py --tag "latte22_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05  --traj_momentum 0.25 --ensemble_rate 0.05 --fast_ensemble &
CUDA_VISIBLE_DEVICES=7  python latte_noise_optimization_150.py --tag "latte22_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.05  --traj_momentum 0.25 --ensemble_rate 0.1 --fast_ensemble &
CUDA_VISIBLE_DEVICES=4 python latte_noise_optimization_150.py --tag "latte22_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05  --traj_momentum 0.75 --ensemble_rate 0.05 --fast_ensemble &
CUDA_VISIBLE_DEVICES=5  python latte_noise_optimization_150.py --tag "latte22_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.05  --traj_momentum 0.75 --ensemble_rate 0.1 --fast_ensemble


# CUDA_VISIBLE_DEVICES=6 python latte22_noise_optimization_150.py --tag "latte22_" \
#     --recall_timesteps 1 --ensemble 10 --momentum 0.05  --traj_momentum 0.3 --ensemble_rate 0.05 --fast_ensemble

# CUDA_VISIBLE_DEVICES=7  python latte22_noise_optimization_150.py --tag "latte22_" \
#     --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.1 --ensemble_rate 0.1 --fast_ensemble

# CUDA_VISIBLE_DEVICES=6 python latte22_noise_optimization_150.py --tag "latte22_" \
#     --recall_timesteps 1 --ensemble 10 --momentum 0.05  --traj_momentum 0.6 --ensemble_rate 0.05 --fast_ensemble

# CUDA_VISIBLE_DEVICES=7  python latte22_noise_optimization_150.py --tag "latte22_" \
#     --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.2 --ensemble_rate 0.1 --fast_ensemble

# CUDA_VISIBLE_DEVICES=6  python latte22_noise_optimization_150.py --tag "latte22_" \
#     --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.4 --ensemble_rate 0.1 --fast_ensemble

# CUDA_VISIBLE_DEVICES=7  python latte22_noise_optimization_150.py --tag "latte22_" \
#     --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.8 --ensemble_rate 0.1 --fast_ensemble