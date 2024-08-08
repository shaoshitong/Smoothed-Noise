import json

import torch
import os
from PIL import Image
import numpy as np
import math
import csv
import random
import torch.distributed as dist
import hpsv2
import ImageReward as RM

from reward_model.eval_pickscore import PickScore
from reward_model.aesthetic_scorer import AestheticScorer

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from diffusers import DDIMScheduler

from noise_model_svd import NoiseDataset
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from noise_model import Solver
from noise_model_svd import SVD_Solver
from noise_model_embedding import Embedding_Solver
from noise_model_svd_embedding import SVD_Embedding_Solver


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_reward_model(metric_version):
      if metric_version == 'PickScore':
            reward_model = PickScore()

      elif metric_version == 'HPSv2':
            reward_model = hpsv2

      elif metric_version == 'ImageReward':
            reward_model = RM.load("ImageReward-v1.0")

      elif metric_version == 'AES':
            reward_model = AestheticScorer(dtype = torch.float32)
        
      else:
            raise NotImplementedError
      
      return reward_model


if __name__ == '__main__':

      dist.init_process_group(backend='nccl')
      local_rank = dist.get_rank()
      torch.cuda.set_device(local_rank)

      metric_version = 'AES'     # ['PickScore', 'HPSv2', 'AES', 'ImageReward']

      reward_model = get_reward_model(metric_version)

      dtype = torch.float16
      before_score, after_score, positive = 0, 0, 0

      # create the datatset 
      bad_datasets = NoiseDataset(
                 all_file=False,
                 evaluate=False,
                 data_dir="/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/datasets/noise_pairs_SDXL_10_100",
                 prompt_path="/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/prompt2seed_SDXL_10_100.txt")

      # make sure the batch size is 1 to evaluate the sample quality
      sampler = torch.utils.data.distributed.DistributedSampler(bad_datasets)
      loader = DataLoader(bad_datasets, batch_size=1, num_workers=8, sampler=sampler)

      dataset_size = bad_datasets.__len__()

      pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype,
                                                     variant='fp16',
                                                     safety_checker=None, requires_safety_checker=False).to(DEVICE)
      pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)


      w = SVD_Embedding_Solver(pipeline=pipeline, local_rank=local_rank, pretrained=True)


      # evaluate model performance
      for step, (original_noise, optimized_noise, prompt, random_seed) in enumerate(loader):
            original_noise, optimized_noise = original_noise.to(DEVICE), optimized_noise.to(DEVICE)
            
            # path is not none, so you can get the idx and random seed
            random_seed = int(random_seed[0])

            np.random.seed(int(random_seed))
            torch.manual_seed(int(random_seed))
            torch.cuda.manual_seed(int(random_seed))

            # path is none ---> original_img, optimized_img; 
            original_scores, predict_scores = w.generate(
                  original_noise, 
                  optimized=None, 
                  reward_model=reward_model,
                  prompt=prompt[0],
                  save_postfix=None,
                  save_pic=None,
                  idx=step,
                  metric_version=metric_version)

            before_score += original_scores
            after_score += predict_scores

            # print(f'seed:{random_seed},  prompt:{prompt}')
            print(f'prompt:{prompt}')
            print(f'origin_score:{original_scores},  optim_score:{predict_scores}')

            if predict_scores > original_scores:
                  positive += 1

      print(f'positive ratio = {(positive / dataset_size)*100}%')
      print(f'original score = {before_score / dataset_size}')
      print(f'optim score = {after_score / dataset_size}')

      
      dist.destroy_process_group()