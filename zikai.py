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
from diffusers import StableDiffusionXLPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = pipe.to(device=device)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0
alpha_schedule = pipe.scheduler.alphas_cumprod.to(device)
sigma_schedule = 1 - pipe.scheduler.alphas_cumprod.to(device)
    
from torchvision import transforms
trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(1024)]) 
    
input = Image.open("./original_image.jpg")
input = trans(input).to(device) * 2 - 1
input = input.unsqueeze(0)
pipe.vae.to(dtype=torch.float32)
input = pipe.vae.config.scaling_factor * pipe.vae.encode(input).latent_dist.sample()
noise = torch.randn_like(input)
add_noise_input = pipe.scheduler.add_noise(input, noise, pipe.scheduler.timesteps[-50].unsqueeze(0))

# input = (pipe.vae.decode(add_noise_input / pipe.vae.config.scaling_factor).sample + 1) / 2
# # input = (add_noise(input) + 1)/2
# input = input.squeeze(0)
# input = transforms.ToPILImage()(input)
# input.save("add_noise.jpg")
# exit(-1)

prompt = ["In slow motion, a dessert fork gently presses into the center of the cake. As the fork goes deeper, the outer crust begins to crack"]
(
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
) = pipe.encode_prompt(prompt=prompt[0], device=device)

add_time_ids = pipe._get_add_time_ids(
    (pipe.default_sample_size * pipe.vae_scale_factor, pipe.default_sample_size * pipe.vae_scale_factor),
    (0, 0),
    (pipe.default_sample_size * pipe.vae_scale_factor, pipe.default_sample_size * pipe.vae_scale_factor),
    dtype=prompt_embeds.dtype,
    text_encoder_projection_dim=int(pooled_prompt_embeds.shape[-1]),
)
added_cond_kwargs = {"text_embeds": pooled_prompt_embeds.to(device), "time_ids": add_time_ids.to(device)}
with torch.no_grad():
    print(add_noise_input.shape, prompt_embeds.shape, pooled_prompt_embeds.shape, add_time_ids.shape)
    denoised_input = pipe.unet(add_noise_input.half(), pipe.scheduler.timesteps[-50].unsqueeze(0).to(device), 
                            encoder_hidden_states=prompt_embeds.to(device),
                            added_cond_kwargs=added_cond_kwargs,
                            cross_attention_kwargs=None).sample
    beta_prod_t = sigma_schedule[pipe.scheduler.timesteps[-50].long().unsqueeze(0).to(device)]
    alpha_prod_t = alpha_schedule[pipe.scheduler.timesteps[-50].long().unsqueeze(0).to(device)]
    latents = (add_noise_input - beta_prod_t ** (0.5) * denoised_input) / alpha_prod_t ** (0.5)
    input = (pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample + 1) / 2
    # input = (add_noise(input) + 1)/2
    input = input.squeeze(0)
    input = transforms.ToPILImage()(input)
    input.save("add_noise.jpg")
