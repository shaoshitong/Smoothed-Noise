CUDA_VISIBLE_DEVICES=6 python modelscope_t2v_noise_optimization_compbench.py --tag "modelscope_final_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025 --noise_type "uniform" --preference_type 0

CUDA_VISIBLE_DEVICES=6 python modelscope_t2v_noise_optimization_compbench.py --tag "modelscope_final_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025 --noise_type "uniform" --preference_type 1

CUDA_VISIBLE_DEVICES=6 python modelscope_t2v_noise_optimization_compbench.py --tag "modelscope_final_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025 --noise_type "uniform" --preference_type 2

CUDA_VISIBLE_DEVICES=6 python modelscope_t2v_noise_optimization_compbench.py --tag "modelscope_final_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025 --noise_type "uniform" --preference_type 3

CUDA_VISIBLE_DEVICES=6 python modelscope_t2v_noise_optimization_compbench.py --tag "modelscope_final_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025 --noise_type "uniform" --preference_type 4

CUDA_VISIBLE_DEVICES=6 python modelscope_t2v_noise_optimization_compbench.py --tag "modelscope_final_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025 --noise_type "uniform" --preference_type 5

CUDA_VISIBLE_DEVICES=6 python modelscope_t2v_noise_optimization_compbench.py --tag "modelscope_final_" \
    --recall_timesteps 1 --ensemble 30 --momentum 0.25  --traj_momentum 0.01 --ensemble_rate 0.025 --noise_type "uniform" --preference_type 6