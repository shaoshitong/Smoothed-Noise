import torch
import requests
import os
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np

from reward_model.eval_pickscore import PickScore
import csv

device = torch.device('cuda')


def load_prompt(path):
    prompts = []
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[1] == "caption":
                continue
            prompts.append(row[1])
    prompts = prompts[0:101]
    tmp_prompt_list = []
    for prompt in prompts:
        if prompt != "":
            tmp_prompt_list.append(prompt)
    prompts = tmp_prompt_list
    return prompts

def sample(prompt, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    mse_loss_list = []
    for i in tqdm(range(start_step, num_inference_steps)):
        t = pipe.scheduler.timesteps[i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            mse_loss_list.append(torch.nn.MSELoss()(noise_pred_uncond, noise_pred_text).item())
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
    # Post-processing
    with torch.no_grad():
        images = pipe.decode_latents(latents.detach())
        images = pipe.numpy_to_pil(images)

    return images, mse_loss_list

if __name__ == '__main__':

    reward_model = PickScore()

    # Load a pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    # Set up a DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    guidance_scale = 3.5
    inference_num = 30
    prompt = 'snow wind and dog'

    x = []
    y = []

    for random_seed in range(100):
        print(f'seed: {random_seed}')
        np.random.seed(int(random_seed))
        torch.manual_seed(int(random_seed))
        torch.cuda.manual_seed(int(random_seed))
        generator = torch.manual_seed(random_seed)
        start_latents = torch.randn(1, 4, 64, 64, generator=generator).to(device)

        g_image, mse_loss_list = sample(prompt, start_latents=start_latents, num_inference_steps=inference_num, guidance_scale=guidance_scale)
        g_image = g_image[0].resize((256, 256))
        after_rewards, optimized_scores = reward_model.calc_probs(prompt, g_image)

        g_image.save(f'/data/bailc/kbqa/Diffusion_Classifier/Classifier-Explore/exp2/g_image_seed_{random_seed}.png')
        print(f'mse_list: {sum(mse_loss_list[:5])/5}, pick_score: {optimized_scores}')

        x.append(sum(mse_loss_list)/len(mse_loss_list))
        y.append(optimized_scores.item())

    h = []
    for a,b in zip(x,y):
        h.append([a,b])
    print(h)
    # print(x)
    # print(y)