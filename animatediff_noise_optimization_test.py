import torch, os
from tqdm import tqdm
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
from utils.pipeline_animatediff_test import AnimateDiffPipeline_GN
from utils.pipeline_animatediff_sdxl import AnimateDiffSDXLPipeline_GN
from diffusers.utils import export_to_gif, export_to_video

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



# pipe.recall_timesteps = 1
# pipe.ensemble = 1
# pipe.pre_free_init_enabled = True
# pipe.fast_ensemble = False
# local_path = "./result_v2.gif"
# output = pipe(
#     prompt="Trump was shot in the left ear during the presidential campaign, yet he still pumped his fist up to lift the crowd",    # negative_prompt="",
#     num_inference_steps=20,
#     guidance_scale=7,
#     width=1024,
#     height=1024,
#     num_frames=16,
# )
# frames = output.frames[0]
# export_to_gif(frames, local_path)


pipe.recall_timesteps = 1
pipe.ensemble = 50
pipe.pre_free_init_enabled = False
pipe.fast_ensemble = False
pipe.ensemble_rate = 0.025
pipe.ensemble_guidance_scale = True
# prompts = [
#     "A young woman in a yellow sweater uses VR glasses, sitting on the shore of a pond on a background of dark waves. A strongwind develops her hair, the sun's rays are reflected from the water.",
#     "A bustling futuristic cityscape at night with neon lights, flying cars, and towering skyscrapers.",
#     "A serene countryside scene with rolling green hills, a small river, and a cozy cottage with smoke coming from the chimney.",
#     "A majestic dragon flying over a medieval castle perched on a cliff, with the sun setting in the background.",
#     "A beautiful underwater scene with colorful coral reefs, various types of fish, and a sunken pirate ship.",
#     "A vibrant street market in a small village in India, with colorful stalls, people in traditional clothing, and spices in the air."
# ]

prompts = [
    "Spiderman is surfing",
    "Yellow and black tropical fish dart through the sea",
    "An epic tornado attacking above aglowing city at night",
    "Slow pan upward of blazing oak fire in an indoor fireplace",
    "a cat wearing sunglasses and working as a lifeguard at pool",
    "A dog in astronaut suit and sunglasses floating in space"
]

for i, prompt in enumerate(prompts):
    local_path = f"./result_rgs_{i}.gif"
    output = pipe(
        prompt=prompt,    # negative_prompt="",
        num_inference_steps=50,
        guidance_scale=7,
        width=512,
        height=512,
        num_frames=16,
    )
    frames = output.frames[0]
    export_to_gif(frames, local_path)

