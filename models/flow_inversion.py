from tqdm import tqdm
from dataclasses import replace
import numpy as np
import torch
import torch.nn.functional as F
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_core.model.transformer.model import LTXModel
from ltx_core.types import LatentState
from .flow_match import FlowMatchScheduler


@torch.inference_mode()
def humo_flowinvert(model, flow_scheduler: FlowMatchScheduler, X0_src, source_input, target_input, start_step, end_step, cfg_scale, model_size='17B', dtype=torch.bfloat16):
    device = X0_src.device
    dimensions = torch.numel(X0_src)

    ref_latents = source_input['y'][0][4:, -1:]

    with torch.autocast('cuda', dtype):

        vinit = model([torch.cat([X0_src, ref_latents], dim=1)], flow_scheduler.timesteps[-1].unsqueeze(0).to(device=device), **source_input)[0]
        fwd_noise = vinit[:,:-1] + X0_src
        noise_norm = torch.norm(fwd_noise.reshape(1, dimensions), dim=1)
        fwd_noise = fwd_noise / noise_norm * np.sqrt(dimensions)

        for progress_id, timestep in enumerate(tqdm(flow_scheduler.timesteps)):
            if progress_id < start_step:
                continue
            
            sigma_i = flow_scheduler.sigmas[progress_id]
            Zt_src = (1-sigma_i) * X0_src + sigma_i * fwd_noise
            if progress_id == start_step:
                Zt_tar = Zt_src.clone()

            Zt_src = torch.cat([Zt_src, ref_latents], dim=1)
            Zt_tar = torch.cat([Zt_tar, ref_latents], dim=1)
            
            timestep = timestep.unsqueeze(0).to(device=device)

            vp_target = model([Zt_tar], timestep, **target_input)[0]
            if cfg_scale > 1:
                vp_source = model([Zt_tar], timestep, **source_input)[0]
                vp = vp_source + cfg_scale * (vp_target - vp_source)
            else:
                vp = vp_target
            
            vp = vp[:,:-1]
            Zt_tar = Zt_tar[:,:-1]

            if progress_id <= end_step:
                vq = model([Zt_src], timestep, **source_input)[0]
                vq = vq[:,:-1]
                Zt_src = Zt_src[:,:-1]
                vtar = vp - vq
                pred_src_noise = Zt_src + (1-sigma_i) * vq

                next_sigma = flow_scheduler.sigmas[progress_id+1] if progress_id+1 < len(flow_scheduler.sigmas) else 0

                
                next_noise = (1-sigma_i) * pred_src_noise + sigma_i * torch.randn_like(pred_src_noise)
                noise_norm = torch.norm(next_noise.reshape(1, dimensions), dim=1)
                next_noise = next_noise / noise_norm * np.sqrt(dimensions)
            
                Zt_tar = Zt_tar + (next_sigma - sigma_i) * vtar + (sigma_i - next_sigma) * X0_src + next_sigma * next_noise - sigma_i * fwd_noise
                fwd_noise = next_noise

            else:
                Zt_tar = flow_scheduler.step(vp, flow_scheduler.timesteps[progress_id], Zt_tar)

    return Zt_tar



@torch.inference_mode()
def ltx_flowinvert(
    model: LTXModel, 
    src_video_state: LatentState,
    src_audio_state: LatentState,
    v_context_src: torch.Tensor,
    v_context_tar: torch.Tensor,
    a_context_src: torch.Tensor,
    a_context_tar: torch.Tensor,
    cfg_scale,
    start_step,
    end_step,
    sigmas,
    dtype=torch.bfloat16,
):
    v_src = src_video_state.latent
    a_src = src_audio_state.latent
    video_dimensions = torch.numel(v_src)
    audio_dimensions = torch.numel(a_src)

    ori_video_src = modality_from_latent_state(src_video_state, v_context_src, sigmas[-2])
    ori_audio_src = modality_from_latent_state(src_audio_state, v_context_src, sigmas[-2])
    ori_video_pred, ori_audio_pred = model(ori_video_src, ori_audio_src, perturbations=None)
    video_noise = v_src + ori_video_pred
    audio_noise = a_src + ori_audio_pred

    vt_tar = (1-sigmas[start_step]) * v_src + sigmas[start_step] * video_noise
    at_tar =  (1-sigmas[start_step]) * a_src + sigmas[start_step] * audio_noise

    video_tar = src_video_state.clone()
    audio_tar = src_audio_state.clone()
    video_tar = replace(video_tar, latent=vt_tar)
    audio_tar = replace(audio_tar, latent=at_tar)

    for step_idx, sigma in enumerate(tqdm(sigmas[:-1])):
        if step_idx < start_step:
            continue
        sigma_next = sigmas[step_idx + 1]
        dt = sigma_next - sigma

        vt_src = (1-sigma) * src_video_state.latent + sigma * video_noise
        at_src = (1-sigma) * src_audio_state.latent + sigma * audio_noise
        
        video_src_state = replace(src_video_state, latent=vt_src)
        audio_src_state = replace(src_audio_state, latent=at_src)
        video_tar_state = replace(src_video_state, latent=video_tar.latent)
        audio_tar_state = replace(src_audio_state, latent=audio_tar.latent)

        pos_video_src = modality_from_latent_state(video_src_state, v_context_src, sigma)
        pos_audio_src = modality_from_latent_state(audio_src_state, a_context_src, sigma)

        pos_video_tar = modality_from_latent_state(video_tar_state, v_context_tar, sigma)
        pos_audio_tar = modality_from_latent_state(audio_tar_state, a_context_tar, sigma)
        neg_video_tar = modality_from_latent_state(video_tar_state, v_context_src, sigma)
        neg_audio_tar = modality_from_latent_state(audio_tar_state, a_context_src, sigma)

        pos_video_tar_pred, pos_audio_tar_pred = model(pos_video_tar, pos_audio_tar, perturbations=None)
        if cfg_scale > 1:
            neg_video_tar_pred, neg_audio_tar_pred = model(neg_video_tar, neg_audio_tar, perturbations=None)
            vp = neg_video_tar_pred + cfg_scale * (pos_video_tar_pred - neg_video_tar_pred)
            ap = neg_audio_tar_pred + cfg_scale * (pos_audio_tar_pred - neg_audio_tar_pred)
        else:
            vp = pos_video_tar_pred
            ap = pos_audio_tar_pred

        if step_idx <= end_step:
            pos_video_src_pred, pos_audio_src_pred = model(pos_video_src, pos_audio_src, perturbations=None)
            vtar = vp - pos_video_src_pred
            atar = ap - pos_audio_src_pred

            if True:
                pred_video_noise = vt_src + (1-sigma) * pos_video_src_pred
                pred_audio_noise = at_src + (1-sigma) * pos_audio_src_pred

                next_video_noise = (1-sigma) * pred_video_noise + sigma * torch.randn_like(pred_video_noise)
                video_noise_norm = torch.norm(next_video_noise.reshape(1, video_dimensions), dim=1)
                next_video_noise = next_video_noise / video_noise_norm * np.sqrt(video_dimensions)

                next_audio_noise = (1-sigma) * pred_audio_noise + sigma * torch.randn_like(pred_audio_noise)
                audio_noise_norm = torch.norm(next_audio_noise.reshape(1, audio_dimensions), dim=1)
                next_audio_noise = next_audio_noise / audio_noise_norm * np.sqrt(audio_dimensions)
            else:
                next_video_noise = torch.randn_like(video_noise)
                next_audio_noise = torch.randn_like(audio_noise)

            video_latent = (video_tar.latent + (vtar - v_src) * dt + sigma_next * next_video_noise - sigma * video_noise).to(dtype)
            audio_latent = (audio_tar.latent + (atar - a_src) * dt + sigma_next * next_audio_noise - sigma * audio_noise).to(dtype)

            video_noise = next_video_noise
            audio_noise = next_audio_noise
        else:
            video_latent = (video_tar.latent.to(torch.float32) + vp.to(torch.float32) * dt).to(dtype)
            audio_latent = (audio_tar.latent.to(torch.float32) + ap.to(torch.float32) * dt).to(dtype)

        video_tar = replace(video_tar, latent=video_latent)
        audio_tar = replace(audio_tar, latent=audio_latent)

    return video_tar, audio_tar