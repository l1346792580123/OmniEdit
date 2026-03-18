import os
from os.path import join
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
import subprocess
import argparse
import json
from omegaconf import OmegaConf
from dataclasses import replace
import math
import numpy as np
import cv2
import imageio.v3 as iio
import librosa
import soundfile
import torch
from einops import rearrange
from safetensors.torch import load_file
from ltx_pipelines.utils import ModelLedger
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.video_vae import TilingConfig
from ltx_pipelines.utils.constants import LTX_2_PARAMS, LTX_2_3_PARAMS
from models.t5 import T5EncoderModel
from models.wanvae import WanVAE
from models.audio_processor import AudioProcessor
from models.flow_match import FlowMatchScheduler
from models.wanmodel import rope_params
from models.humo import HuMo
from models.ltx_utils import mel_spectrogram, build_noise, video_patchify, video_unpatchify, audio_patchify, audio_unpatchify, decode_video_audio, encode_prompt
from models.flow_inversion import humo_flowinvert, ltx_flowinvert

def combine_av(video_path, audio_path, output_path):
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-i", "%s"%video_path,
        "-i", "%s"%audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "%s"%output_path
    ]

    subprocess.run(cmd, check=True)

def preprocess_lipsync_data(video_path, audio_path, num_frames, device, downsample_ratio=8, max_size=921600, sr=16000, max_batch=3):
    import decord

    vr = decord.VideoReader(video_path)
    orig_fps = vr.get_avg_fps()
    total_frames = len(vr)
    have_audio = True

    try:
        audio_input, _ = librosa.load(video_path, sr=sr)
    except:
        have_audio = False
    
    tar_audio_input, _ = librosa.load(audio_path, sr=sr)
    tar_audio_duration = tar_audio_input.shape[0] / sr

    batch_duration = num_frames / orig_fps
    batch_audio_size = int(batch_duration * sr)
    ori_height, ori_width = vr[0].shape[:2]

    indices = np.linspace(0, total_frames-1, round(total_frames/orig_fps*orig_fps))
    video_duration = len(indices) / orig_fps
    duration = min(video_duration, tar_audio_duration)
    num_batch = min(int(duration/batch_duration), max_batch)
    duration = num_batch * batch_duration

    if have_audio:
        audio_input = audio_input[:num_batch*batch_audio_size].reshape(num_batch, batch_audio_size)
    else:
        audio_input = None
    tar_audio_input = tar_audio_input[:num_batch*batch_audio_size].reshape(num_batch, batch_audio_size)
    indices = indices[:num_batch*num_frames]

    video = vr.get_batch(indices).asnumpy()
    scale = min(1, math.sqrt(max_size / (ori_height*ori_width)))
    height, width = int((ori_height*scale) // (downsample_ratio*2) * (downsample_ratio*2)), int((ori_width*scale) // (downsample_ratio*2) * (downsample_ratio*2))
    video = np.stack([cv2.resize(frame, (width, height)) for frame in video])
    video = (torch.from_numpy(video/255.).float().permute(3,0,1,2) - 0.5) / 0.5

    video = rearrange(video, "c (b f) h w -> b c f h w", b=num_batch, f=num_frames)
    video = video.unsqueeze(1).to(device)

    seq_len = int(width * height * ((num_frames+7)//4) / ((downsample_ratio*2)**2))

    return video, audio_input, tar_audio_input, seq_len, sr, orig_fps

def lipsync(cfg):
    device = torch.device('cuda')
    mixed_precision = cfg['mixed_precision']
    wan_vae_path = cfg['wan_vae_path']
    wan_text_path = cfg['wan_text_path']
    wan_tokenizer_path = cfg['wan_tokenizer_path']
    audio_encoder_path = cfg['audio_encoder_path']
    checkpoint_path = cfg['checkpoint_path']
    dtype = torch.bfloat16 if mixed_precision == 'bf16' else torch.float32
    model_size = '17B' if cfg['model']['dim'] == 5120 else '1.7B'
    num_frames = cfg['num_frames']
    start_step = cfg['start_step']
    end_step = cfg['end_step']
    cfg_scale = cfg['cfg_scale']

    with torch.device('meta'):
        model = HuMo(**cfg['model'])
    
    d = cfg['model']['dim'] // cfg['model']['num_heads']
    model.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)

    state_dict = {}
    for pth in checkpoint_path:
        if pth.endswith('safetensors'):
            state_dict.update(load_file(pth, device='cpu'))
        elif pth.endswith('pth'):
            state_dict.update(torch.load(pth,map_location='cpu'))
    
    model.load_state_dict(state_dict, assign=True, strict=True)
    model.to(dtype=dtype, device=device)

    text_encoder = T5EncoderModel(
        text_len=cfg['model']['text_len'],
        dtype=dtype,
        device=torch.device('cpu'),
        checkpoint_path=wan_text_path,
        tokenizer_path=wan_tokenizer_path,
        shard_fn=None)
    
    vae = WanVAE(vae_pth=wan_vae_path, device=device)

    audio_processor = AudioProcessor(16000, 25, audio_encoder_path, 'all', None, None, '', device)

    flow_scheduler = FlowMatchScheduler(**cfg['scheduler'])

    for video_path, audio_path, prompt in zip(cfg['video_paths'], cfg['audio_paths'], cfg['prompts']):
        video_prefix = video_path.split('/')[-1].split('.')[0]
        audio_prefix = audio_path.split('/')[-1].split('.')[0]
        video_name = '%s_%s.mp4'%(video_prefix,audio_prefix)

        video, src_audio, tar_audio, seq_len, sr, target_fps = preprocess_lipsync_data(video_path, audio_path, num_frames, device)

        ret = []
        with torch.no_grad():
            text_encoder.model.to(device)
            context = text_encoder(prompt, device)
            text_encoder.model.to('cpu')
            torch.cuda.empty_cache()

            for idx, (video_part, tar_audio_part) in enumerate(zip(video, tar_audio)):
                batch_size = video_part.shape[0]
                assert batch_size == 1 and video_part.shape[2] == num_frames

                latents = vae.encode([video_part[i] for i in range(batch_size)])[0]
                zero_latents = vae.encode([torch.zeros_like(video_part[i]) for i in range(batch_size)])[0]
                ref_latents = vae.encode([video_part[i][:,0:1] for i in range(batch_size)])[0]

                lat_f, lat_h, lat_w = latents.shape[1:]

                tar_audio_emb = audio_processor.get_audio_feat(tar_audio_part, target_fps, num_frames)
                if src_audio is None:
                    src_audio_emb = torch.zeros_like(tar_audio_emb)
                else:
                    src_audio_emb = audio_processor.get_audio_feat(src_audio[idx], target_fps, num_frames)
                
                zero_audio_pad = torch.zeros(1, 1, *src_audio_emb.shape[2:]).to(src_audio_emb.device)
                src_audio_emb = torch.cat([src_audio_emb, zero_audio_pad], dim=1)
                tar_audio_emb = torch.cat([tar_audio_emb, zero_audio_pad], dim=1)
                msk = torch.ones(4, lat_f+1, lat_h, lat_w, device=device)
                msk[:,:-1] = 0
                y_c = torch.cat([zero_latents, ref_latents], dim=1)
                y = torch.concat([msk, y_c])

                src_input = {
                    'clip_fea': None,
                    'context': context,
                    'seq_len': seq_len,
                    'y': [y],
                    'audio_input': src_audio_emb,
                }

                tar_input = {
                    'clip_fea': None,
                    'context': context,
                    'seq_len': seq_len,
                    'y': [y],
                    'audio_input': tar_audio_emb,
                }

                output_latents = humo_flowinvert(model, flow_scheduler, latents, src_input, tar_input, start_step, end_step, cfg_scale, model_size)
                pred_video = ((vae.decode([output_latents])[0].permute(1,2,3,0).cpu().numpy() + 1.)*127.5).astype(np.uint8)
                ret.append(pred_video)
            
            ret = np.concatenate(ret, axis=0)
            iio.imwrite('tmp.mp4',ret,fps=target_fps,codec="libx264",pixelformat="yuv420p",bitrate="20M",macro_block_size=1)
            soundfile.write('tmp.wav', tar_audio.reshape(-1), sr)
            combine_av('tmp.mp4', 'tmp.wav', video_name)



def preprocess_avedit_data(video_path, device, dtype=torch.bfloat16, num_frames=121, downsample_ratio=32, max_size=921600, sr=16000):
    import decord

    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    audio_input, _ = librosa.load(video_path, sr=sr, mono=False)

    batch_duration = num_frames / fps
    batch_audio_size = round(batch_duration * sr)

    ori_height, ori_width = vr[0].shape[:2]
    indices = np.arange(num_frames)
    video = vr.get_batch(indices).asnumpy()
    scale = min(1, math.sqrt(max_size / (ori_height*ori_width)))
    height, width = int(((ori_height*scale) // downsample_ratio) * downsample_ratio), int(((ori_width*scale) // downsample_ratio) * downsample_ratio)
    video = np.stack([cv2.resize(frame, (width, height)) for frame in video])
    video = (torch.from_numpy(video/255.).float().permute(3,0,1,2) - 0.5) / 0.5
    video = rearrange(video, "c (b f) h w -> b c f h w", b=1, f=num_frames)
    video = video.to(device=device, dtype=dtype)

    audio_input = torch.from_numpy(audio_input[:, :batch_audio_size]).float().to(device=device)
    mel, stft_spec = mel_spectrogram(audio_input)
    mel = mel.unsqueeze(0).to(device=device, dtype=dtype)

    return video, mel, fps

def avedit(cfg):
    device = torch.device('cuda')
    model_path = cfg['model_path']
    gemma_root = cfg['gemma_root']
    num_frames = cfg['num_frames']
    start_step = cfg['start_step']
    end_step = cfg['end_step']
    cfg_scale = cfg['cfg_scale']
    model_version = cfg['model_version']
    params = LTX_2_3_PARAMS if model_version == '2.3' else LTX_2_PARAMS
    dtype = torch.bfloat16
    tiling_config = TilingConfig.default()

    model_ledger = ModelLedger(dtype, torch.device('cpu'), model_path, gemma_root)

    dit = model_ledger.transformer().velocity_model
    video_encoder = model_ledger.video_encoder()
    video_decoder = model_ledger.video_decoder()
    audio_encoder = model_ledger.audio_encoder()
    audio_decoder = model_ledger.audio_decoder()
    text_encoder = model_ledger.text_encoder()
    embeddings_processor = model_ledger.gemma_embeddings_processor()
    vocoder = model_ledger.vocoder()
    

    sigmas = LTX2Scheduler().execute(steps=params.num_inference_steps).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for video_path, src_prompt, tar_prompt in zip(cfg['video_paths'], cfg['src_prompts'], cfg['tar_prompts']):
            src_prefix = video_path.split('/')[-1].split('.')[0]

            frames, mel, fps = preprocess_avedit_data(video_path, device, dtype, num_frames)
            num_frames, height, width = frames.shape[2:]

            text_encoder.to(device)
            embeddings_processor.to(device)
            ctx_src, ctx_tar = encode_prompt([src_prompt, tar_prompt], text_encoder, embeddings_processor)
            text_encoder.to('cpu')
            embeddings_processor.to('cpu')
            torch.cuda.empty_cache()

            v_context_src, a_context_src = ctx_src.video_encoding, ctx_src.audio_encoding
            v_context_tar, a_context_tar = ctx_tar.video_encoding, ctx_tar.audio_encoding

            video_state, audio_state, video_latent_shape, audio_latent_shape = build_noise(num_frames, width, height, fps, generator=None, device=device, dtype=dtype)

            video_encoder.to(device)
            audio_encoder.to(device)
            src_video_latent = video_patchify(video_encoder(frames))
            src_video_state = replace(video_state, latent=src_video_latent)
            src_audio_latent = audio_patchify(audio_encoder(mel))
            src_audio_state = replace(audio_state, latent=src_audio_latent)
            video_encoder.to('cpu')
            audio_encoder.to('cpu')
            torch.cuda.empty_cache()

            dit.to(device)
            tar_video_state, tar_audio_state = ltx_flowinvert(dit, src_video_state, src_audio_state, v_context_src, v_context_tar, a_context_src, a_context_tar, cfg_scale, start_step, end_step, sigmas)
            dit.to('cpu')
            torch.cuda.empty_cache()

            tar_video_latent = video_unpatchify(tar_video_state.latent, video_latent_shape.frames, video_latent_shape.height, video_latent_shape.width)
            tar_audio_latent = audio_unpatchify(tar_audio_state.latent, audio_latent_shape.channels, audio_latent_shape.mel_bins)

            video_decoder.to(device)
            audio_decoder.to(device)
            vocoder.to(device)
            ret_video, ret_audio = decode_video_audio(tar_video_latent, tar_audio_latent, video_decoder, audio_decoder, vocoder, tiling_config)
            video_decoder.to('cpu')
            audio_decoder.to('cpu')
            vocoder.to('cpu')
            torch.cuda.empty_cache()

            iio.imwrite('tmp.mp4',ret_video,fps=fps,codec="libx264",pixelformat="yuv420p",bitrate="20M",macro_block_size=1)
            soundfile.write('tmp.wav', ret_audio.T, 24000)
            
            tar_name = '%s_%s.mp4'%(src_prefix, tar_prompt)
            combine_av('tmp.mp4', 'tmp.wav', tar_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/avedit.yaml')

    arg = parser.parse_args()
    cfg = OmegaConf.load(arg.config)


    if cfg.task == 'lipsync':
        lipsync(cfg)
    elif cfg.task == 'avedit':
        avedit(cfg)