import inspect, copy
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMInverseScheduler,
    PNDMScheduler,
)
from diffusers import AnimateDiffPipeline


class AnimateDiffPipeline_GN(AnimateDiffPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        motion_adapter,
        scheduler,
        feature_extractor = None,
        image_encoder = None,
    ):
        super(AnimateDiffPipeline_GN, self).__init__(
                vae = vae,
                text_encoder = text_encoder,
                tokenizer = tokenizer,
                unet = unet,
                motion_adapter = motion_adapter,
                scheduler = scheduler,
                feature_extractor = feature_extractor,
                image_encoder = image_encoder,  
        )
        self.recall_timesteps = 1
        self.ensemble = 1
    
    @torch.no_grad()
    def __call__(self,
        prompt: Union[str, List[str]] = None,
        num_frames: Optional[int] = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        *args,**kwargs):
        
        _height = height or self.unet.config.sample_size * self.vae_scale_factor
        _width = width or self.unet.config.sample_size * self.vae_scale_factor
        _num_videos_per_prompt = 1
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
        
        _prompt_embeds, _negative_prompt_embeds = self.encode_prompt(
            prompt,
            _device,
            _num_videos_per_prompt,
            _do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds = kwargs.get("prompt_embeds", None),
            negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None),
            lora_scale=_text_encoder_lora_scale,
            clip_skip= kwargs.get("clip_skip", None),
        )
        if _do_classifier_free_guidance:
            _prompt_embeds = torch.cat([_negative_prompt_embeds, _prompt_embeds])
        _emergency_scheduler = copy.deepcopy(self.scheduler)
        _emergency_scheduler.set_timesteps(num_inference_steps, device=_device)
        _timesteps = _emergency_scheduler.timesteps
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

        added_cond_kwargs = None
        extra_step_kwargs = self.prepare_extra_step_kwargs(kwargs.get("generator", None), kwargs.get("eta", 0.))
        if self.free_init_enabled:
            _latents, _timesteps = self._apply_free_init(
                _latents, 0, num_inference_steps, _device, _latents.dtype, kwargs.get("generator", None)
            ) # Apply FreeInit
        _inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                                    subfolder='scheduler')
        _inverse_scheduler.set_timesteps(num_inference_steps, device=_device)
        _lasting_t = _timesteps[0]
        _prev_lasting_t = _timesteps[0] -_emergency_scheduler.config.num_train_timesteps // _emergency_scheduler.num_inference_steps
        _optim_steps = self.recall_timesteps
        
        for i in range(_optim_steps):
            if self.ensemble == 1:
                _latent_model_input = torch.cat([_latents] * 2) if _do_classifier_free_guidance else _latents # Forward DDIM
                _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                _noise_pred = self.unet(
                    _latent_model_input,
                    _lasting_t,
                    encoder_hidden_states=_prompt_embeds,
                    cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                if _do_classifier_free_guidance:
                    _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                    _noise_pred = _noise_pred_uncond + guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                _latents = _emergency_scheduler.step(_noise_pred, _lasting_t, _latents, **extra_step_kwargs).prev_sample

                _latent_model_input = torch.cat([_latents] * 2) if _do_classifier_free_guidance else _latents  # Inverse DDIM
                _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                _noise_pred = self.unet(
                    _latent_model_input,
                    _prev_lasting_t,
                    encoder_hidden_states=_prompt_embeds,
                    cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                if _do_classifier_free_guidance:
                    _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                    _noise_pred = _noise_pred_uncond + guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                _inv_extra_step_kwargs = copy.deepcopy(extra_step_kwargs)
                _inv_extra_step_kwargs.pop("eta")
                _latents = _inverse_scheduler.step(_noise_pred, _lasting_t, _latents, return_dict=False)[0]
            else:
                results = []
                for j in range(self.ensemble):
                    _latents_n = torch.randn_like(_latents) * 0.1 + _latents
                    _latent_model_input = torch.cat([_latents_n] * 2) if _do_classifier_free_guidance else _latents_n # Forward DDIM
                    _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                    _noise_pred = self.unet(
                        _latent_model_input,
                        _lasting_t,
                        encoder_hidden_states=_prompt_embeds,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    if _do_classifier_free_guidance:
                        _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                        _noise_pred = _noise_pred_uncond + guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                    _latents_n = _emergency_scheduler.step(_noise_pred, _lasting_t, _latents_n, **extra_step_kwargs).prev_sample
                    _latent_model_input = torch.cat([_latents_n] * 2) if _do_classifier_free_guidance else _latents_n  # Inverse DDIM
                    _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                    _noise_pred = self.unet(
                        _latent_model_input,
                        _prev_lasting_t,
                        encoder_hidden_states=_prompt_embeds,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    if _do_classifier_free_guidance:
                        _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                        _noise_pred = _noise_pred_uncond + guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                    _inv_extra_step_kwargs = copy.deepcopy(extra_step_kwargs)
                    _inv_extra_step_kwargs.pop("eta")
                    _latents_n = _inverse_scheduler.step(_noise_pred, _lasting_t, _latents_n, return_dict=False)[0]
                    results.append(_latents_n)
                _latents = torch.stack(results,0).mean(0)
                
        print("Successfully Generate Optimium Noise")
        if kwargs.get("prompt_2", None) is not None:
            prompt = kwargs.get("prompt_2", None) # Second Prompt
        
        return super().__call__(latents = _latents,
                                prompt = prompt,
                                num_frames = num_frames,
                                height = height,
                                width = width,
                                num_inference_steps = num_inference_steps,
                                guidance_scale = guidance_scale,
                                negative_prompt = negative_prompt,
                                num_videos_per_prompt = num_videos_per_prompt,
                                *args,**kwargs)
        
        
