import inspect, copy
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from tqdm import tqdm
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMInverseScheduler,
    PNDMScheduler,
)
from diffusers import AnimateDiffSDXLPipeline
from diffusers.pipelines.animatediff.pipeline_animatediff_sdxl import retrieve_timesteps


class AnimateDiffSDXLPipeline_GN(AnimateDiffSDXLPipeline):
    def __init__(self,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        motion_adapter,
        scheduler,
        image_encoder = None,
        feature_extractor = None,
        force_zeros_for_empty_prompt = True,
    ):
        super(AnimateDiffSDXLPipeline_GN, self).__init__(
                vae = vae,
                text_encoder = text_encoder,
                text_encoder_2 = text_encoder_2,
                tokenizer = tokenizer,
                tokenizer_2 = tokenizer_2,
                unet = unet,
                motion_adapter = motion_adapter,
                scheduler = scheduler,
                image_encoder = image_encoder,
                feature_extractor = feature_extractor,
                force_zeros_for_empty_prompt = force_zeros_for_empty_prompt,
        )
        self.recall_timesteps = 1
        self.ensemble = 1
        self.ensemble_rate = 0.1
        self.pre_num_inference_steps = 50
        self.pre_free_init_enabled = False
        self.fast_ensemble = True
        self.momentum = 0.01
    
    @torch.no_grad()
    def __call__(self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None, # Additional
        num_frames: Optional[int] = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None, # Additional
        sigmas: List[float] = None, # Additional
        denoising_end: Optional[float] = None, # Additional
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None, # Additional
        num_videos_per_prompt: Optional[int] = 1,
        *args,**kwargs):
        
        _height = height or self.unet.config.sample_size * self.vae_scale_factor
        _width = width or self.unet.config.sample_size * self.vae_scale_factor
        _num_videos_per_prompt = 1

        _original_size = kwargs.get("original_size", None) or (_height, _width)
        _target_size = kwargs.get("target_size", None) or (_height, _width)
        
        if prompt is not None and isinstance(prompt, str):
            _batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            _batch_size = len(prompt)
        else:
            _batch_size = kwargs.get("prompt_embeds", None).shape[0]
        _device = self._execution_device
        
        _text_encoder_lora_scale = (
            kwargs.get("cross_attention_kwargs", None).get("scale", None) if kwargs.get("cross_attention_kwargs", None) is not None else None
        )
        _do_classifier_free_guidance = guidance_scale > 1

        (
            _prompt_embeds,
            _negative_prompt_embeds,
            _pooled_prompt_embeds,
            _negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=_device,
            num_videos_per_prompt=_num_videos_per_prompt,
            do_classifier_free_guidance=_do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=kwargs.get("prompt_embeds", None),
            negative_prompt_embeds=kwargs.get("negative_prompt_embeds", None),
            pooled_prompt_embeds=kwargs.get("pooled_prompt_embeds", None),
            negative_pooled_prompt_embeds=kwargs.get("negative_pooled_prompt_embeds", None),
            lora_scale=_text_encoder_lora_scale,
            clip_skip=kwargs.get("clip_skip", None),
        )

        pre_num_inference_steps = self.pre_num_inference_steps
        _emergency_scheduler = copy.deepcopy(self.scheduler)
        _timesteps, _num_inference_steps = retrieve_timesteps(
            _emergency_scheduler, pre_num_inference_steps, _device, timesteps, sigmas
        )
        _set_num_inference_steps = _num_inference_steps
        _num_channels_latents = self.unet.config.in_channels
        _latents = self.prepare_latents(
            _batch_size * _num_videos_per_prompt,
            _num_channels_latents,
            num_frames,
            _height,
            _width,
            _prompt_embeds.dtype,
            _device,
            kwargs.get("generator", None),
            kwargs.get("latents", None)
        )
        
        # added_cond_kwargs = None
        extra_step_kwargs = self.prepare_extra_step_kwargs(kwargs.get("generator", None), kwargs.get("eta", 0.))
        _add_text_embeds = _pooled_prompt_embeds
        if self.text_encoder_2 is None:
            _text_encoder_projection_dim = int(_pooled_prompt_embeds.shape[-1])
        else:
            _text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        _add_time_ids = self._get_add_time_ids(
            _original_size,
            kwargs.get("crops_coords_top_left", (0,0)),
            _target_size,
            dtype=_prompt_embeds.dtype,
            text_encoder_projection_dim=_text_encoder_projection_dim,
        )
        if kwargs.get("negative_original_size", None) is not None and kwargs.get("negative_target_size", None) is not None:
            _negative_add_time_ids = self._get_add_time_ids(
                kwargs.get("negative_original_size", None),
                kwargs.get("negative_crops_coords_top_left", (0,0)),
                kwargs.get("negative_target_size", None),
                dtype=_prompt_embeds.dtype,
                text_encoder_projection_dim=_text_encoder_projection_dim,
            )
        else:
            _negative_add_time_ids = _add_time_ids
        
        
        if _do_classifier_free_guidance:
            _add_text_embeds = torch.cat([_negative_pooled_prompt_embeds, _add_text_embeds], dim=0)
            _add_time_ids = torch.cat([_negative_add_time_ids, _add_time_ids], dim=0)
            _prompt_embeds = torch.cat([_negative_prompt_embeds, _prompt_embeds])
        
        _prompt_embeds = _prompt_embeds.to(_device)
        _add_text_embeds = _add_text_embeds.to(_device)
        _add_time_ids = _add_time_ids.to(_device).repeat(_batch_size * _num_videos_per_prompt, 1)
        
        if (
            denoising_end is not None
            and isinstance(denoising_end, float)
            and denoising_end > 0
            and denoising_end < 1
        ):
            _discrete_timestep_cutoff = int(
                round(
                    _emergency_scheduler.config.num_train_timesteps
                    - (denoising_end * _emergency_scheduler.config.num_train_timesteps)
                )
            )
            _num_inference_steps = len(list(filter(lambda ts: ts >= _discrete_timestep_cutoff, _timesteps)))
            _timesteps = _timesteps[:_num_inference_steps]

        _timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            _guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(_batch_size * _num_videos_per_prompt)
            _timestep_cond = self.get_guidance_scale_embedding(
                _guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=_device, dtype=_latents.dtype)

        _inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                                    subfolder='scheduler')
        _inverse_scheduler.set_timesteps(pre_num_inference_steps, device=_device)
        _lasting_t = _timesteps[0]
        _prev_lasting_t = _timesteps[0] -_emergency_scheduler.config.num_train_timesteps // _emergency_scheduler.num_inference_steps
        print(_lasting_t, _prev_lasting_t)
        _optim_steps = self.recall_timesteps
        _added_cond_kwargs = {"text_embeds": _add_text_embeds, "time_ids": _add_time_ids}
        for i in range(_optim_steps):
            if self.ensemble == 1:
                _latent_model_input = torch.cat([_latents] * 2) if _do_classifier_free_guidance else _latents # Forward DDIM
                _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                _noise_pred = self.unet(
                    _latent_model_input,
                    _lasting_t,
                    encoder_hidden_states=_prompt_embeds,
                    timestep_cond=_timestep_cond,
                    cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    added_cond_kwargs=_added_cond_kwargs,
                    return_dict=False,
                )[0]
                if _do_classifier_free_guidance:
                    _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                    _noise_pred = _noise_pred_uncond + guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                _latents = _emergency_scheduler.step(_noise_pred, _lasting_t, _latents, **extra_step_kwargs, return_dict=False)[0]

                _latent_model_input = torch.cat([_latents] * 2) if _do_classifier_free_guidance else _latents  # Inverse DDIM
                _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _prev_lasting_t)
                _noise_pred = self.unet(
                    _latent_model_input,
                    _prev_lasting_t,
                    encoder_hidden_states=_prompt_embeds,
                    timestep_cond=_timestep_cond,
                    cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    added_cond_kwargs=_added_cond_kwargs,
                    return_dict=False,
                )[0]
                if _do_classifier_free_guidance:
                    _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                    _noise_pred = _noise_pred_uncond - 1. * (_noise_pred_text - _noise_pred_uncond)
                _inv_extra_step_kwargs = copy.deepcopy(extra_step_kwargs)
                _inv_extra_step_kwargs.pop("eta")
                _latents = _inverse_scheduler.step(_noise_pred, _lasting_t, _latents, return_dict=False)[0]
            else:
                results = []
                for j in tqdm(range(self.ensemble)):
                    if self.fast_ensemble:  
                        _latents = self.momentum * results[-1] + (1 - self.momentum) * _latents if len(results)>0 else _latents
                        _latents_n = torch.randn_like(_latents) * self.ensemble_rate * (1 - self.momentum) ** (j+1) + _latents
                    else:
                        _latents_n = torch.randn_like(_latents) * self.ensemble_rate + _latents
                    _latent_model_input = torch.cat([_latents_n] * 2) if _do_classifier_free_guidance else _latents_n # Forward DDIM
                    _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                    _noise_pred = self.unet(
                        _latent_model_input,
                        _lasting_t,
                        encoder_hidden_states=_prompt_embeds,
                        timestep_cond=_timestep_cond,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                        added_cond_kwargs=_added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    if _do_classifier_free_guidance:
                        _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                        _noise_pred = _noise_pred_uncond - 1. * (_noise_pred_text - _noise_pred_uncond)
                    _latents_n = _emergency_scheduler.step(_noise_pred, _lasting_t, _latents_n, **extra_step_kwargs, return_dict=False)[0]
                    _latent_model_input = torch.cat([_latents_n] * 2) if _do_classifier_free_guidance else _latents_n  # Inverse DDIM
                    _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                    _noise_pred = self.unet(
                        _latent_model_input,
                        _prev_lasting_t,
                        encoder_hidden_states=_prompt_embeds,
                        timestep_cond=_timestep_cond,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                        added_cond_kwargs=_added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    if _do_classifier_free_guidance:
                        _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                        _noise_pred = _noise_pred_uncond + guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                    _inv_extra_step_kwargs = copy.deepcopy(extra_step_kwargs)
                    _inv_extra_step_kwargs.pop("eta")
                    _latents_n = _inverse_scheduler.step(_noise_pred, _lasting_t, _latents_n, return_dict=False)[0]
                    results.append(_latents_n)
                if self.fast_ensemble:
                    _latents = results[-1]
                else:
                    _latents = torch.stack(results,0).mean(0)
                
        print("Successfully Generate Optimium Noise")
        if kwargs.get("prompt_2", None) is not None:
            prompt = kwargs.get("prompt_2", None) # Second Prompt
        
        return super().__call__(    latents = _latents,
                                    prompt = prompt,
                                    num_frames = num_frames,
                                    height = height,
                                    width = width,
                                    num_inference_steps = num_inference_steps,
                                    guidance_scale = guidance_scale,
                                    negative_prompt = negative_prompt,
                                    num_videos_per_prompt = num_videos_per_prompt,
                                    *args,**kwargs)
        
        
