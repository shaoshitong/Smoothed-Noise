# CUDA_VISIBLE_DEVICES=4 python animatediff_xl_noise_optimization.py --tag "g7_" \
#     --recall_timesteps 1 --ensemble 140 --momentum 0.15 --ensemble_rate 0.05 # --fast_ensemble

CUDA_VISIBLE_DEVICES=4 python animatediff_xl_noise_optimization.py --tag "g7_" \
    --recall_timesteps 1 --ensemble 140 --momentum 0.15 --ensemble_rate 0.025 # --fast_ensemble

# CUDA_VISIBLE_DEVICES=4 python animatediff_xl_noise_optimization.py --tag "g7_" \
#     --recall_timesteps 1 --ensemble 140 --momentum 0.15 --ensemble_rate 0.075 --fast_ensemble

# CUDA_VISIBLE_DEVICES=4 python animatediff_xl_noise_optimization.py --tag "g7_" \
#     --recall_timesteps 1 --ensemble 140 --momentum 0.15 --ensemble_rate 0.1 --fast_ensemble