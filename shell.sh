CUDA_VISIBLE_DEVICES=6 python animatediff_xl_noise_optimization.py --tag "g7_animatediff_xl_final2" \
    --recall_timesteps 1 --ensemble 20 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.1 --fast_ensemble

CUDA_VISIBLE_DEVICES=0 python animatediff_xl_noise_optimization.py --tag "g7_animatediff_xl_final2" \
    --recall_timesteps 1 --ensemble 20 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025 --noise_type "uniform"

CUDA_VISIBLE_DEVICES=3 python animatediff_xl_noise_optimization.py --tag "g7_animatediff_xl_final2" \
    --recall_timesteps 1 --ensemble 6 --momentum 0.05  --traj_momentum 0.05 --ensemble_rate 0.05 --fast_ensemble

CUDA_VISIBLE_DEVICES=2 python animatediff_xl_noise_optimization.py --tag "g7_animatediffv3_final2" \
    --recall_timesteps 1 --ensemble 1 --momentum 0.05  --traj_momentum 0.05 --ensemble_rate 0.05

CUDA_VISIBLE_DEVICES=1 python animatediff_xl_noise_optimization.py --tag "g7_animatediffv3_final2" \
    --recall_timesteps 0 --ensemble 1 --momentum 0.05  --traj_momentum 0.05 --ensemble_rate 0.05