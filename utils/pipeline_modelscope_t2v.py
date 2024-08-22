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
from diffusers import TextToVideoSDPipeline


class ModelScopeT2V_GN(TextToVideoSDPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
    ):
        super(ModelScopeT2V_GN, self).__init__(
                vae = vae,
                text_encoder = text_encoder,
                tokenizer = tokenizer,
                unet = unet,
                scheduler = scheduler 
        )
        self.recall_timesteps = 1
        self.ensemble = 1
        self.ensemble_rate = 0.1
        self.pre_num_inference_steps = 50
        self.fast_ensemble = False
        self.momentum = 0.
        self.traj_momentum = 0.05
        self.ensemble_guidance_scale = False
        self.noise_type = "uniform"
    
    @torch.no_grad()
    def __call__(self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = 16,
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
        _inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                                    subfolder='scheduler')
        _inverse_scheduler.set_timesteps(num_inference_steps, device=_device)
        _lasting_t = _timesteps[0]
        _prev_lasting_t = _timesteps[0] -_emergency_scheduler.config.num_train_timesteps // _emergency_scheduler.num_inference_steps
        _optim_steps = self.recall_timesteps
        
        for i in range(_optim_steps):
            if self.ensemble == 1:
                if self.ensemble_guidance_scale:
                    guidance_scale = guidance_scale
                    rand = torch.randn(1).item()
                positive_guidance_scale = guidance_scale if not self.ensemble_guidance_scale else guidance_scale + rand
                negative_guidance_scale = guidance_scale if not self.ensemble_guidance_scale else guidance_scale + rand
            
                _latent_model_input = torch.cat([_latents] * 2) if _do_classifier_free_guidance else _latents # Forward DDIM
                _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                _noise_pred = self.unet(
                    _latent_model_input,
                    _lasting_t,
                    encoder_hidden_states=_prompt_embeds,
                    cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    return_dict=False,
                )[0]
                if _do_classifier_free_guidance:
                    _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                    _noise_pred = _noise_pred_uncond + positive_guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                # Additional
                _bsz, _channel, _frames, _width, _height = _latents.shape
                _latents = _latents.permute(0, 2, 1, 3, 4).reshape(_bsz * _frames, _channel, _width, _height)
                _noise_pred = _noise_pred.permute(0, 2, 1, 3, 4).reshape(_bsz * _frames, _channel, _width, _height)
                _latents = _emergency_scheduler.step(_noise_pred, _lasting_t, _latents, **extra_step_kwargs).prev_sample
                _latents = _latents[None, :].reshape(_bsz, _frames, _channel, _width, _height).permute(0, 2, 1, 3, 4)
                _latent_model_input = torch.cat([_latents] * 2) if _do_classifier_free_guidance else _latents  # Inverse DDIM
                _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                _noise_pred = self.unet(
                    _latent_model_input,
                    _prev_lasting_t,
                    encoder_hidden_states=_prompt_embeds,
                    cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                ).sample
                if _do_classifier_free_guidance:
                    _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                    _noise_pred = _noise_pred_uncond + negative_guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                _inv_extra_step_kwargs = copy.deepcopy(extra_step_kwargs)
                _inv_extra_step_kwargs.pop("eta")
                # Additional
                _latents = _latents.permute(0, 2, 1, 3, 4).reshape(_bsz * _frames, _channel, _width, _height)
                _noise_pred = _noise_pred.permute(0, 2, 1, 3, 4).reshape(_bsz * _frames, _channel, _width, _height)
                _latents = _inverse_scheduler.step(_noise_pred, _lasting_t, _latents, return_dict=False)[0]
                _latents = _latents[None, :].reshape(_bsz, _frames, _channel, _width, _height).permute(0, 2, 1, 3, 4)
            else:
                results = []
                _prev_latents = _latents.clone()
                _prev_prev_latents = _latents.clone()
                
                for j in range(self.ensemble):
                    if self.noise_type == "uniform":
                        print("pass uniform")
                        additional_noise = (torch.rand_like(_latents) - 0.5) * 2 * (3 ** (1/2))
                    elif self.noise_type == "truncated_gaussian":
                        additional_noise = torch.randn_like(_latents).clip_(-1, 1)
                    else:
                        additional_noise = torch.randn_like(_latents)
                    if self.fast_ensemble:
                        print("pass fast-ensemble")
                        _prev_prev_latents = _prev_latents
                        _prev_latents = _latents
                        _latents = results[-1] * (1-self.traj_momentum) + self.traj_momentum * (1-self.traj_momentum) * _prev_latents + self.traj_momentum * self.traj_momentum * _prev_prev_latents if len(results)>0 else _latents
                        _latents_n = additional_noise * self.ensemble_rate * (1 - self.momentum) ** (j+1) + _latents
                    else:
                        _latents_n = additional_noise * self.ensemble_rate + _latents
                    if self.ensemble_guidance_scale:
                        guidance_scale = guidance_scale
                        rand = torch.randn(1).item()
                    positive_guidance_scale = guidance_scale if not self.ensemble_guidance_scale else guidance_scale + rand
                    negative_guidance_scale = 1.0 if not self.ensemble_guidance_scale else 1.0 + rand
                    
                    _latent_model_input = torch.cat([_latents_n] * 2) if _do_classifier_free_guidance else _latents_n # Forward DDIM
                    _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                    _noise_pred = self.unet(
                        _latent_model_input,
                        _lasting_t,
                        encoder_hidden_states=_prompt_embeds,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    ).sample
                    if _do_classifier_free_guidance:
                        _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                        _noise_pred = _noise_pred_uncond + positive_guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                    # Additional
                    _bsz, _channel, _frames, _width, _height = _latents_n.shape
                    _latents_n = _latents_n.permute(0, 2, 1, 3, 4).reshape(_bsz * _frames, _channel, _width, _height)
                    _noise_pred = _noise_pred.permute(0, 2, 1, 3, 4).reshape(_bsz * _frames, _channel, _width, _height)
                    _latents_n = _emergency_scheduler.step(_noise_pred, _lasting_t, _latents_n, **extra_step_kwargs).prev_sample
                    _latents_n = _latents_n[None, :].reshape(_bsz, _frames, _channel, _width, _height).permute(0, 2, 1, 3, 4)
                    
                    _latent_model_input = torch.cat([_latents_n] * 2) if _do_classifier_free_guidance else _latents_n  # Inverse DDIM
                    _latent_model_input = _emergency_scheduler.scale_model_input(_latent_model_input, _lasting_t)
                    _noise_pred = self.unet(
                        _latent_model_input,
                        _prev_lasting_t,
                        encoder_hidden_states=_prompt_embeds,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    ).sample
                    if _do_classifier_free_guidance:
                        _noise_pred_uncond, _noise_pred_text = _noise_pred.chunk(2)
                        _noise_pred = _noise_pred_uncond + negative_guidance_scale * (_noise_pred_text - _noise_pred_uncond)
                    _inv_extra_step_kwargs = copy.deepcopy(extra_step_kwargs)
                    _inv_extra_step_kwargs.pop("eta")
                    # Additional                  
                    _latents_n = _latents_n.permute(0, 2, 1, 3, 4).reshape(_bsz * _frames, _channel, _width, _height)
                    _noise_pred = _noise_pred.permute(0, 2, 1, 3, 4).reshape(_bsz * _frames, _channel, _width, _height)
                    _latents_n = _inverse_scheduler.step(_noise_pred, _lasting_t, _latents_n, return_dict=False)[0]
                    _latents_n = _latents_n[None, :].reshape(_bsz, _frames, _channel, _width, _height).permute(0, 2, 1, 3, 4)
                    
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
                                    *args,**kwargs)
        
        
