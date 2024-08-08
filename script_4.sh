CUDA_VISIBLE_DEVICES=7 python animatediff_xl_noise_optimization.py --tag "g71_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.15 --ensemble_rate 0.05

CUDA_VISIBLE_DEVICES=7 python animatediff_xl_noise_optimization.py --tag "g71_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.15 --ensemble_rate 0.025

CUDA_VISIBLE_DEVICES=7 python animatediff_xl_noise_optimization.py --tag "g71_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.15 --ensemble_rate 0.075

CUDA_VISIBLE_DEVICES=7 python animatediff_xl_noise_optimization.py --tag "g71_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.15 --ensemble_rate 0.1


CUDA_VISIBLE_DEVICES=7 python animatediff_xl_noise_optimization.py --tag "g72_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.15 --ensemble_rate 0.05

CUDA_VISIBLE_DEVICES=7 python animatediff_xl_noise_optimization.py --tag "g72_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.15 --ensemble_rate 0.025

CUDA_VISIBLE_DEVICES=7 python animatediff_xl_noise_optimization.py --tag "g72_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.15 --ensemble_rate 0.075

CUDA_VISIBLE_DEVICES=7 python animatediff_xl_noise_optimization.py --tag "g72_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.15 --ensemble_rate 0.1