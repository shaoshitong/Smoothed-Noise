import torch, os
from tqdm import tqdm
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
from utils.pipeline_animatediff import AnimateDiffPipeline_GN
from utils.pipeline_animatediff_sdxl import AnimateDiffSDXLPipeline_GN
from diffusers.utils import export_to_gif, export_to_video
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Smoothed Path Optimization")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--method",
        type=str,
        default="sdxl",
        choices=["sdv1.5", "sdxl"],
        help="The name of the bese model to use.",
    )
    parser.add_argument(
        "--recall_timesteps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="gaussian",
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--traj_momentum",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--ensemble_rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--fast_ensemble",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    prompt_file = pd.read_csv("/home/shaoshitong/project/NoiseModel/VideoGoldenNoise/Captions_ChronoMagic-Bench-150.csv",
                            usecols=[0,1])
    prompts = list(prompt_file["name"])
    filenames = list(prompt_file["videoid"])
    method = args.method
    if method == "sdxl":
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16)
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipe = AnimateDiffSDXLPipeline_GN.from_pretrained(
            model_id,
            motion_adapter=adapter,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        pipe.recall_timesteps = args.recall_timesteps
        pipe.ensemble = args.ensemble
        pipe.momentum = args.momentum
        pipe.traj_momentum = args.traj_momentum
        pipe.ensemble_rate = args.ensemble_rate
        pipe.fast_ensemble = args.fast_ensemble
        pipe.noise_type = args.noise_type
        print("recall_timesteps", pipe.recall_timesteps, 
          "ensemble", pipe.ensemble, 
          "momentum", pipe.momentum, 
          "traj_momentum",pipe.traj_momentum,
          "ensemble_rate", pipe.ensemble_rate, 
          "fast_ensemble", pipe.fast_ensemble,
          "noise_type",pipe.noise_type)
    else:
        # Load the motion adapter
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)
        # adapter = MotionAdapter.from_pretrained("vladmandic/animatediff-v3").to(dtype=torch.float16,device=torch.device("cuda"))
        # load SD 1.5 based finetuned model
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        pipe = AnimateDiffPipeline_GN.from_pretrained(model_id, motion_adapter=adapter).to(dtype=torch.float16,device=torch.device("cuda"))
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1
        )
        pipe.scheduler = scheduler
        pipe.recall_timesteps = args.recall_timesteps
        pipe.ensemble = args.ensemble
        pipe.momentum = args.momentum
        pipe.traj_momentum = args.traj_momentum
        pipe.ensemble_rate = args.ensemble_rate
        pipe.fast_ensemble = args.fast_ensemble
        pipe.noise_type = args.noise_type
    print("recall_timesteps", pipe.recall_timesteps, 
          "ensemble", pipe.ensemble, 
          "momentum", pipe.momentum, 
          "traj_momentum",pipe.traj_momentum,
          "ensemble_rate", pipe.ensemble_rate, 
          "fast_ensemble", pipe.fast_ensemble,
          "noise_type",pipe.noise_type)
    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    # pipe.enable_xformers_memory_efficient_attention()
    root_path = os.path.join("/data/shaoshitong/Chronomagic/", f"{args.method}", "total_150", args.tag + f"_en_{args.ensemble}_fast_{args.fast_ensemble}_mom_{args.momentum}_trajmom{args.traj_momentum}_enr_{args.ensemble_rate}_recall_{args.recall_timesteps}")

    if not os.path.exists(root_path):
        os.makedirs(root_path)


    for filename, prompt in tqdm(zip(filenames,prompts)):
        local_path = os.path.join(root_path,filename + ".mp4")
        if not os.path.exists(local_path):
            if args.method == "sdxl":
                output =  pipe(
                    prompt=prompt,
                    negative_prompt="",
                    num_inference_steps=20,
                    guidance_scale=8,
                    width=1024,
                    height=1024,
                    num_frames=16,
                )
            else:
                output = pipe(
                    prompt=prompt,
                    # negative_prompt="",
                    num_inference_steps=50,
                    guidance_scale=7,
                    num_frames=16,
                )
            frames = output.frames[0]
            export_to_video(frames, local_path)