import os
import json
from typing import Union
from PIL import Image

from . import utils


environ_root = os.environ.get('HPS_ROOT')
root_path = os.path.expanduser('~/.cache/hpsv2') if environ_root == None else environ_root
name = 'hpsv2'
url = 'https://github.com/tgxs002/HPSv2'
os.environ['NO_PROXY'] = 'huggingface.co'

# Acquire available models
# available_models = utils.get_available_models()

# Model Abbreviations Dict
model_ab_dict = {
        'CM': 'ChilloutMix',
        'Cog2': 'CogView2',
        'DALLE-mini': 'DALL·E mini',
        'DALLE': 'DALL·E 2',
        'DF-IF': 'DeepFloyd-XL',
        'DL': 'Dreamlike Photoreal 2.0',
        'Deliberate': 'Deliberate',
        'ED': 'Epic Diffusion',
        'FD': 'FuseDream',
        'LDM': 'Latent Diffusion',
        'Laf': 'LAFITE',
        'MM': 'MajicMix Realistic',
        'OJ': 'Openjourney',
        'RV': 'Realistic Vision',
        'SDXL-base-0.9': 'SDXL Base 0.9',
        'SDXL-refiner-0.9': 'SDXL Refiner 0.9',
        'SDXL-base-1.0': 'SDXL Base 1.0',
        'SDXL-refiner-1.0': 'SDXL Refiner 1.0',
        'VD': 'Versatile Diffusion',
        'VQD': 'VQ-Diffusion',
        'VQGAN': 'VQGAN + CLIP',
        'glide': 'GLIDE',
        'sdv1': 'Stable Diffusion v1.4',
        'sdv2': 'Stable Diffusion v2.0'
    }

def evaluate(imgs_path: str) -> None:
    """Evaluate images generated by any text-to-image model based on benchmark prompts

    Args:
        img_path (str): path to generated image
    """
    utils.download_benchmark_prompts()
    data_path = os.path.join(root_path, 'datasets/benchmark')
    
    from . import evaluation as eval
    eval.evaluate(mode="benchmark", data_path=data_path, root_dir=imgs_path)


def evaluate_benchmark(model_id: str) -> None:
    """Evaluate benchmark images generated by the example models based on benchmark prompts

    Args:
        model_id (str): Name of example model (one of available_models)
    """
    utils.download_benchmark_prompts()

    if model_id in available_models:
        utils.download_benchmark_images(model_id)
    else:
        raise ValueError(f'The banchmark data of {model_id} model is not available.')
    
    try:
        i = list(model_ab_dict.values()).index(model_id)
        model_id = list(model_ab_dict.keys())[i]
    except ValueError:
        print('Input model not in model dict.')
        if model_id not in model_ab_dict.keys():
            pass
    
    imgs_path = os.path.join(root_path, f'datasets/benchmark/benchmark_imgs/{model_id}')
    data_path = os.path.join(root_path, 'datasets/benchmark')
    
    from . import evaluation as eval
    eval.evaluate(mode="benchmark", data_path=data_path, root_dir=imgs_path)

def score(imgs_path: Union[list, str, Image.Image], prompt: str) -> list:
    """Score the image and prompt

    Args:
        imgs_path (Union[list, str, Image.Image]): paths to generated image(s)
        prompt (str): corresponding prompt

    Returns:
        list: matching scores for images and prompt
    """

    from . import img_score as scr
    res = scr.score(imgs_path, prompt)
    return res

def benchmark_prompts(style: str = 'all') -> Union[dict, list]:
    """Get benchmark prompts of certain style

    Args:
        style (str, optional): Defaults to 'all'.

    Raises:
        ValueError: Style is illegal

    Returns:
        Union[dict, list]: return {} if style == 'all', else return [] 
    """
    styles = ['anime', 'concept-art', 'paintings', 'photo']
    
    if (style != 'all') and style not in styles:
        raise ValueError('Style is illegal. You must choose from "all", "anime", "concept-art", "paintings", "photo".')
    
    utils.download_benchmark_prompts()
    
    if style == 'all':
        res = {}
        for sty in styles:
            style_path = os.path.join(root_path, f'datasets/benchmark/{sty}.json')
            with open(style_path) as f:
                prompts = json.load(f)
            res[sty] = prompts
        return res
    
    else:
        style_path = os.path.join(root_path, f'datasets/benchmark/{style}.json')
        with open(style_path) as f:
            prompts = json.load(f)
        return prompts
