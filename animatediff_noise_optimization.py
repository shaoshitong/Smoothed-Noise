import os
import math
import random
import numpy as np
import accelerate
from tqdm import tqdm
from torch import nn
import copy, cv2, PIL

import tempfile
import torch

from torch.nn.functional import mse_loss
from PIL import Image
from reward_model.eval_pickscore import PickScore
from torch.utils.data import DataLoader, Dataset
from diffusers.optimization import get_scheduler
from torch import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.pipeline_animatediff import AnimateDiffPipeline_GN
from diffusers import MotionAdapter, AnimateDiffPipeline
from diffusers.schedulers import DDIMScheduler
import torch.distributed as dist
from reward_model.eval_pickscore import PickScore
import PIL.Image
import PIL.ImageOps
from diffusers.utils.import_utils import (
    BACKENDS_MAPPING,
    is_opencv_available,
)
from single_lisa import LISADiffusion
from diffusers.models.normalization import AdaGroupNorm

def get_avi_files(directory):
    results = []
    for root, dirs, files in os.walk(directory):
        avi_files = [f for f in files if f.endswith('.avi')]
        results = results + avi_files
    return results


def export_to_video(
    video_frames, 
    output_video_path: str = None, 
    fps: int = 10,
    w = 240, 
    h = 320
) -> str:
    
    import cv2
    
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _, _, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    lw = random.randint(0, 360 - w)
    lh = random.randint(0, 360 - h)
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        print(img.mean())
        img = cv2.resize(img, (360, 360))
        img = img[lh:lh+h,lw:lw+w,:]
        video_writer.write(img)
    return output_video_path

DEVICE = torch.device("cuda" if torch.cuda else "cpu")
NUM_INFERENCE = 10
random_seed = 42
if __name__ == '__main__':
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2").to(dtype=torch.float16,device=torch.device("cuda"))
    # load SD 1.5 based finetuned model
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline_GN.from_pretrained(model_id, motion_adapter=adapter).to(dtype=torch.float16,device=torch.device("cuda"))
    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1
    )
    pipe.scheduler = scheduler
    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()

    pipe.recall_timesteps = 1
    pipe.ensemble = 1
    
    with open("after_ucf_prompt.txt", "r+") as f:
        m = f.readlines()
        m = [i.strip() for i in m]
    prompts = []
    local_prompt = []
    for i in m:
        if i == "":
            prompts.append(copy.deepcopy(local_prompt))
            local_prompt = []
        else:
            local_prompt.append(i)
    with open("number_count.txt", "r+") as f:
        m = f.readlines()
        m = [int(i.strip()) for i in m]
    numbers = m

    with open("ucf_prompt.txt", "r+") as f:
        m = f.readlines()
        m = [i.strip() for i in m]
    titles = m
    
    root_path = '/data/shaoshitong/UCF101/UCF_fake_v2/'
    filenames = get_avi_files('/data/shaoshitong/UCF101/UCF-101/')
    count = 0
    for i, number in tqdm(enumerate(numbers)):
        for _ in range(number):
            current_prompt = prompts[i][random.randint(0, len(prompts[i])-1)]
            current_prompt = f"The title is {titles[i]}." + current_prompt
            video_frames = pipe(current_prompt, num_inference_steps=50).frames
            _path = root_path + filenames[count].replace(".avi",".mp4")
            export_to_video(video_frames[0], _path)
            count +=1