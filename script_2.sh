CUDA_VISIBLE_DEVICES=3 python animatediff_xl_noise_optimization_150.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.05 --ensemble_rate 0.05 --fast_ensemble --method sdv1.5 &
CUDA_VISIBLE_DEVICES=4 python animatediff_xl_noise_optimization_150.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.05 --ensemble_rate 0.025 --fast_ensemble --method sdv1.5 &
CUDA_VISIBLE_DEVICES=5 python animatediff_xl_noise_optimization_150.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.05 --ensemble_rate 0.075 --fast_ensemble --method sdv1.5 &
CUDA_VISIBLE_DEVICES=6 python animatediff_xl_noise_optimization_150.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.05 --ensemble_rate 0.1 --fast_ensemble --method sdv1.5

CUDA_VISIBLE_DEVICES=3 python animatediff_xl_noise_optimization_150.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05 --ensemble_rate 0.05 --fast_ensemble --method sdv1.5 &
CUDA_VISIBLE_DEVICES=4 python animatediff_xl_noise_optimization_150.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05 --ensemble_rate 0.025 --fast_ensemble --method sdv1.5 &
CUDA_VISIBLE_DEVICES=5 python animatediff_xl_noise_optimization_150.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05 --ensemble_rate 0.075 --fast_ensemble --method sdv1.5 &
CUDA_VISIBLE_DEVICES=6 python animatediff_xl_noise_optimization_150.py --tag "g1_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05 --ensemble_rate 0.1 --fast_ensemble --method sdv1.5

CUDA_VISIBLE_DEVICES=3 python animatediff_xl_noise_optimization_150.py --tag "g11_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05 --ensemble_rate 0.05 --fast_ensemble --method sdv1.5 &
CUDA_VISIBLE_DEVICES=4 python animatediff_xl_noise_optimization_150.py --tag "g11_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05 --ensemble_rate 0.025 --fast_ensemble --method sdv1.5 &
CUDA_VISIBLE_DEVICES=5 python animatediff_xl_noise_optimization_150.py --tag "g11_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05 --ensemble_rate 0.075 --fast_ensemble --method sdv1.5 &
CUDA_VISIBLE_DEVICES=6 python animatediff_xl_noise_optimization_150.py --tag "g11_" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05 --ensemble_rate 0.1 --fast_ensemble --method sdv1.5
