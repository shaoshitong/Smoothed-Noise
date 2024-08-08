import torch
from diffusers import KolorsPipeline

pipe = KolorsPipeline.from_pretrained(
    "Kwai-Kolors/Kolors-diffusers", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")
prompt = '一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着"可图"'
image = pipe(
    prompt=prompt,
    negative_prompt="",
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator(pipe.device).manual_seed(66),
).images[0]
image.show()