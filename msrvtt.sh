CUDA_VISIBLE_DEVICES=5 python modelscope_t2v_noise_optimization_msrvtt.py --tag "MSRVTT_FVD" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.1 --fast_ensemble &
CUDA_VISIBLE_DEVICES=4 python modelscope_t2v_noise_optimization_msrvtt.py --tag "MSRVTT_FVD" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025 &
CUDA_VISIBLE_DEVICES=3 python modelscope_t2v_noise_optimization_msrvtt.py --tag "MSRVTT_FVD" \
    --recall_timesteps 1 --ensemble 1 --momentum 0.05  --traj_momentum 0.05 --ensemble_rate 0.05 &
CUDA_VISIBLE_DEVICES=2 python modelscope_t2v_noise_optimization_msrvtt.py --tag "MSRVTT_FVD" \
    --recall_timesteps 0 --ensemble 1 --momentum 0.05  --traj_momentum 0.05 --ensemble_rate 0.05 &
CUDA_VISIBLE_DEVICES=2 python modelscope_t2v_noise_optimization_msrvtt.py --tag "MSRVTT_FVD" \
    --recall_timesteps 1 --ensemble 10 --momentum 0.05  --traj_momentum 0.05 --ensemble_rate 0.05 --fast_ensemble
