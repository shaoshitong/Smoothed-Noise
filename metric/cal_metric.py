import torch
import requests
import os
from PIL import Image
from tqdm.auto import tqdm
from reward_model.eval_pickscore import PickScore
import hpsv2
import ImageReward as RM
from reward_model.aesthetic_scorer import AestheticScorer
import csv


def load_single_image_list(file_path):
    image_list = os.listdir(file_path)
    res = []
    for idx in range(len(image_list)):
        res.append(Image.open(os.path.join(file_path, f'{idx}.png')))
    return res


def load_prompt(path, dataset_version):
    if dataset_version == 'pick_score':
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
    elif dataset_version == 'drawBench':
        prompts = []
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == "Prompts":
                    continue
                prompts.append(row[0])

        prompts = prompts[0:200]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list
    else:
        return None

    return prompts


def load_image(path):
    origin_list = os.listdir(os.path.join(path, 'origin'))
    optim_list = os.listdir(os.path.join(path, 'optim'))
    
    origin_image_list, optim_image_list = [], []
    for idx in range(len(origin_list)):
        # set path_origin to baseline reslut path
        # origin_image_list.append(Image.open(os.path.join(path_origin,'origin',f'{idx}.png')))
        # set path_origin to path
        origin_image_list.append(Image.open(os.path.join(path,'origin',f'{idx}.png')))
    for idx in range(len(optim_list)):
        optim_image_list.append(Image.open(os.path.join(path,'optim',f'{idx}.png')))
    return origin_image_list, optim_image_list

def cal_score(prompt_list, image_list, metric_version):
    prompt_list = prompt_list[:len(image_list)]
    assert len(prompt_list) == len(image_list)
    total_score = 0
    score_list = []
    if metric_version == 'PickScore':
        reward_model = PickScore()
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            _, score = reward_model.calc_probs(prompt, image)
            total_score += score
            score_list.append(score)

    elif metric_version == 'HPSv2':
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            score = hpsv2.score([image], prompt, hps_version="v2.1")
            print(score)
            total_score += score[0]
            score_list.append(score)

    elif metric_version == 'ImageReward':
        reward_model = RM.load("ImageReward-v1.0")
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            score = reward_model.score(prompt, image)
            # print(score)
            total_score += score
            score_list.append(score)

    elif metric_version == 'AES':
        reward_model = AestheticScorer(dtype = torch.float32)
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            score = reward_model(image)
            total_score += score[0]
            score_list.append(score[0])
    else:
        raise NotImplementedError

    return total_score/len(prompt_list), score_list

if __name__ == '__main__':
    #load prompt_list
    dataset_version = 'pick_score'  #['pick_score', 'drawBench']
    metric_version = 'AES'     #['PickScore', 'HPSv2', 'AES', 'ImageReward']

    if dataset_version == 'pick_score':
        prompt_path = '/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/datasets/prompt/test_unique_caption_zh.csv'
    elif dataset_version == 'drawBench':
        prompt_path = './datasets/drawbench.csv'
    else:
        raise NotImplementedError
    prompt_list = load_prompt(prompt_path, dataset_version)

    #load image_list
    # image_dir_path = './exp_data/draw_sdxl_main'
    image_dir_path = '/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/datasets/output_SDXL_10_100'
    print(f'load image from path: {image_dir_path}')


    image_origin, image_optim = load_image(image_dir_path)
    # image_optim = image_origin
    print('loaded image ...')

    # #TODO nips
    # # optim v1 path
    # if dataset_version == 'pick_score':
    #     nips_img_path = './exp_data/baseline/SDXL-50steps-pick-nips'
    # else:
    #     nips_img_path = './exp_data/baseline/SDXL-50steps-draw-nips'
    # nips_img_list = load_single_image_list(nips_img_path)

    #origin metric
    print(f'start cal {metric_version} about origin images')
    origin_score, origin_score_list = cal_score(prompt_list, image_origin, metric_version)

    #optim metric
    print(f'start cal {metric_version} about optim images')
    optim_score, optim_score_list = cal_score(prompt_list, image_optim, metric_version)
    # print(f'optim_score: {optim_score}')

    # nips metric
    # print(f'start cal {metric_version} about nips images')
    # nips_score, nips_score_list = cal_score(prompt_list, nips_img_list, metric_version)

    # print(f'{metric_version}')
    # print(f'origin_score:{origin_score}    nips_score:{nips_score}     optim_score:{optim_score}')


    positive = 0
    for origin,optim in zip(origin_score_list, optim_score_list):
        if origin < optim:
            positive += 1
    print(f'wining rate between origin and optim: {positive/len(optim_score_list)}')
    print(f'origin_score:{origin_score}      optim_score:{optim_score}')


    # positive = 0
    # for origin, optim in zip(origin_score_list, nips_score_list):
    #     if origin < optim:
    #         positive += 1
    # print(f'wining rate between origin and nips: {positive / len(optim_score_list)}')

    # positive = 0
    # for origin, optim in zip(nips_score_list, optim_score_list):
    #     if origin < optim:
    #         positive += 1
    # print(f'wining rate between nips and optim: {positive / len(optim_score_list)}')
    #
    # a,b,c=0,0,0
    # for origin,nips,optim in zip(origin_score_list, nips_score_list, optim_score_list):
    #     if origin > nips and origin > optim:
    #         a += 1
    #     elif nips > origin and nips > optim:
    #         b +=1
    #     elif optim > origin and optim > nips:
    #         c += 1
    #
    # print(a,b,c)