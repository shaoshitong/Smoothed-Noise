import torch, os
from typing import Optional
from tqdm import tqdm
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
from diffusers import (
    AnimateDiffPipeline,
    DiffusionPipeline,
    UNet3DConditionModel,
    LCMScheduler,
    MotionAdapter,
)
from diffusers.utils import export_to_gif, export_to_video
import argparse
import pandas as pd
from accelerate.utils import ProjectConfiguration, set_seed
set_seed(42)
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict

def get_animatediff_pipeline(
    real_variant: Optional[str] = "realvision",
    motion_module_path: str = "guoyww/animatediff-motion-adapter-v1-5-2",
):
    if real_variant is None:
        model_id = "runwayml/stable-diffusion-v1-5"
    elif real_variant == "epicrealism":
        model_id = "emilianJR/epiCRealism"
    elif real_variant == "realvision":
        model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    else:
        raise ValueError(f"Unknown real_variant {real_variant}")

    adapter = MotionAdapter.from_pretrained(
        motion_module_path, torch_dtype=torch.float16
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    scheduler = LCMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        timestep_scaling=4.0,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        beta_start=0.00085,
        beta_end=0.012,
        steps_offset=1,
    )
    pipe.scheduler = scheduler
    pipe.enable_vae_slicing()
    return pipe


def get_modelscope_pipeline():
    model_id = "ali-vilab/text-to-video-ms-1.7b"
    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    import diffusers
    scheduler = LCMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        timestep_scaling=4.0,
    )
    pipe.scheduler = scheduler
    pipe.enable_vae_slicing()

    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="Accelerated Sampling from A Data-centric Perspective")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--tag",
        type=str,
        default="",
    )
    parser.add_argument(
        "--mcm",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    prompt_file = pd.read_csv("./Captions_ChronoMagic-Bench-150.csv",
                            usecols=[0,1])
    prompts = list(prompt_file["name"])
    filenames = list(prompt_file["videoid"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = get_modelscope_pipeline()
    if args.mcm:
        lora = PeftModel.from_pretrained(
        pipeline.unet,
        "yhzhai/mcm",
        subfolder="modelscopet2v-webvid",
        adapter_name="pretrained_lora",
        torch_device="cpu",
        )
        lora.merge_and_unload()
        pipeline.unet = lora
    else:
        lora = UNet3DConditionModel.from_pretrained(
        args.model_path,
        torch_device="cpu")
        unet = lora
        pipeline.unet = unet
    pipeline = pipeline.to(device,dtype=torch.float16)
    
    # enable memory savings
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    # pipe.enable_xformers_memory_efficient_attention()
    root_path = os.path.join("/data/shaoshitong/Chronomagic/", f"accelerated_sampling", "total_150", args.tag + f"_step-{args.num_inference_steps}")

    if not os.path.exists(root_path):
        os.makedirs(root_path)


    for filename, prompt in tqdm(zip(filenames,prompts)):
        local_path = os.path.join(root_path,filename + ".mp4")
        if not os.path.exists(local_path):
            output = pipeline(
                prompt=prompt,
                num_frames=16,
                guidance_scale=1.0,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(42),
            )
            frames = output.frames[0]
            export_to_video(frames, local_path)