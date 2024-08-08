import json

import torch
import os
from PIL import Image
import numpy as np
import math
import csv
import random

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from evaluate_models import get_reward_model
from noise_model_svd import NoiseDataset
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_prob(original_img, 
              optimized_img, 
              reward_model, 
              metric_version):
      
      if metric_version == 'PickScore':    
            before_rewards, original_scores = reward_model.calc_probs(prompt, original_img)
            golden_rewards, optimized_scores = reward_model.calc_probs(prompt, optimized_img)
      elif metric_version == 'HPSv2':
            original_scores = reward_model.score([original_img], prompt, hps_version="v2.1")[0]
            optimized_scores = reward_model.score([optimized_img], prompt, hps_version="v2.1")[0]
      elif metric_version == 'ImageReward':
            original_scores = reward_model.score(prompt, original_img)
            optimized_scores = reward_model.score(prompt, optimized_img)
      elif metric_version == 'AES':
            original_scores = reward_model(original_img)
            optimized_scores = reward_model(optimized_img)
      
      return original_scores, optimized_scores


def evaluate_sample(pipeline, original_noise, optimized_noise, prompt, path):
      random_seed, idx = path.split('/')[-1].split('.')[0].split('_')
      random_seed = int(random_seed)
      idx = int(idx)

      np.random.seed(int(random_seed))
      torch.manual_seed(int(random_seed))
      torch.cuda.manual_seed(int(random_seed))
      
      original_img = pipeline(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=5.5,
                num_inference_steps=50,
                latents=original_noise,
                ratioT=0.9).images[0]

      optimized_img = pipeline(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=5.5,
            num_inference_steps=50,
            latents=optimized_noise,
            ratioT=0.9).images[0]
      
      return original_img, optimized_img, idx, random_seed


if __name__ == '__main__':
      dtype = torch.float16
      before_score, after_score, positive = 0, 0, 0

      metric_version = 'AES'     # ['PickScore', 'HPSv2', 'AES', 'ImageReward']

      reward_model = get_reward_model(metric_version)

      # create the datatset 
      bad_datasets = NoiseDataset(evaluate=True)

      # make sure the batch size is 1 to evaluate the sample quality
      loader = DataLoader(bad_datasets, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

      dataset_size = bad_datasets.__len__()

      pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype,
                                                      variant='fp16',
                                                      safety_checker=None, requires_safety_checker=False).to(DEVICE)

      with open('bad_sample_pick.json', 'w', encoding='utf-8') as f:
            # evaluate samples
            for step, (original_noise, optimized_noise, prompt, path) in enumerate(loader):
                  original_noise, optimized_noise = original_noise.to(DEVICE), optimized_noise.to(DEVICE)

                  original_img, optimized_img, idx, random_seed = evaluate_sample(
                        pipeline,
                        original_noise, 
                        optimized_noise, 
                        prompt, 
                        path[0])

                  original_scores, optimized_scores = calc_prob(original_img, optimized_img, reward_model, metric_version)
                  
                  before_score += original_scores
                  after_score += optimized_scores

                  print(f'seed:{random_seed},  prompt:{prompt}')
                  print(f'origin_score:{original_scores},  optim_score:{optimized_scores}')

                  if optimized_scores > original_scores:
                        positive += 1
                  else:
                        data = {
                              "path": path[0],
                              "orginal_scores": float(original_scores.cpu().numpy()[0]),
                              "optimized_scores": float(optimized_scores.cpu().numpy()[0])
                        }     
                        json.dump(data, f)
      
            print(f'positive ratio = {(positive / dataset_size)*100}%')
            print(f'original score = {before_score / dataset_size}')
            print(f'optim score = {after_score / dataset_size}')