CUDA_VISIBLE_DEVICES=6 python animatediff_xl_noise_optimization.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 50 --momentum 0.15 --ensemble_rate 0.025 --begin 0 --end 8000 &
CUDA_VISIBLE_DEVICES=7 python animatediff_xl_noise_optimization.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 50 --momentum 0.15 --ensemble_rate 0.025 --begin 800 --end 2000

# CUDA_VISIBLE_DEVICES=6 python animatediff_xl_noise_optimization.py --tag "g1_" \
#     --recall_timesteps 1 --ensemble 50 --momentum 0.15 --ensemble_rate 0.025 --begin 1200 --end 1800