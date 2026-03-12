import torch
import einops
from librosa.filters import mel as librosa_mel_fn
from safetensors.torch import load_file
from ltx_core.types import LatentState, VideoPixelShape, VideoLatentShape, AudioLatentShape
from ltx_core.components.patchifiers import VideoLatentPatchifier, AudioPatchifier, get_pixel_coords
from ltx_pipelines.utils.constants import VIDEO_LATENT_CHANNELS, VIDEO_SCALE_FACTORS
from ltx_core.model.video_vae.video_vae import VideoEncoder, VideoDecoder
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.model.audio_vae.audio_vae import AudioDecoder
from ltx_core.model.audio_vae.vocoder import Vocoder
from ltx_core.model.upsampler.model import LatentUpsampler

def encode_prompt(prompts: list[str], text_encoder, embeddings_processor):
    raw_outputs = [text_encoder.encode(p) for p in prompts]
    results = [ embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]

    return results

def video_patchify(latents, _patch_size=(1,1,1)):
    latents = einops.rearrange(
        latents,
        "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
        p1=_patch_size[0],
        p2=_patch_size[1],
        p3=_patch_size[2],
    )

    return latents

def video_unpatchify(latents, frames, height, width, _patch_size=(1,1,1)):

    patch_grid_frames = frames // _patch_size[0]
    patch_grid_height = height // _patch_size[1]
    patch_grid_width = width // _patch_size[2]

    latents = einops.rearrange(
        latents,
        "b (f h w) (c p q) -> b c f (h p) (w q)",
        f=patch_grid_frames,
        h=patch_grid_height,
        w=patch_grid_width,
        p=_patch_size[1],
        q=_patch_size[2],
    )

    return latents


def audio_patchify(audio_latents):

    audio_latents = einops.rearrange(
        audio_latents,
        "b c t f -> b t (c f)",
    )

    return audio_latents

def audio_unpatchify(audio_latents, channels=8, mel_bins=16):

    audio_latents = einops.rearrange(
        audio_latents,
        "b t (c f) -> b c t f",
        c=channels,
        f=mel_bins,
    )

    return audio_latents


def build_noise(
    num_frames,
    width,
    height,
    fps=24,
    batch_size=1,
    generator=None,
    device='cuda',
    dtype=torch.bfloat16,
    noise_scale=1.0,
    init_video_latent=None,
    init_audio_latent=None,
):
    video_patchifier = VideoLatentPatchifier(patch_size=1)
    audio_patchifier = AudioPatchifier(patch_size=1)

    pixel_shape = VideoPixelShape(
        batch=batch_size,
        frames=num_frames,
        width=width,
        height=height,
        fps=fps,
    )
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=pixel_shape,
        latent_channels=VIDEO_LATENT_CHANNELS,
        scale_factors=VIDEO_SCALE_FACTORS,
    )
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)

    video_noise = video_patchifier.patchify(torch.randn(*video_latent_shape.to_torch_shape(),generator=generator, device=device, dtype=dtype))
    if init_video_latent is not None:
        video_clean_latents = init_video_latent
    else:
        video_clean_latents = torch.zeros_like(video_noise)
    video_noise = noise_scale * video_noise + (1-noise_scale) * video_clean_latents
    video_denoising_mask = video_patchifier.patchify(torch.ones(*video_latent_shape.mask_shape().to_torch_shape(), device=device, dtype=torch.float32))
    video_latent_coords = video_patchifier.get_patch_grid_bounds(output_shape=video_latent_shape, device=device)
    positions = get_pixel_coords(latent_coords=video_latent_coords, scale_factors=VIDEO_SCALE_FACTORS, causal_fix=True).float()
    positions[:, 0, ...] = positions[:, 0, ...] / fps

    audio_noise = audio_patchifier.patchify(torch.randn(*audio_latent_shape.to_torch_shape(),generator=generator, device=device, dtype=dtype))
    if init_audio_latent is not None:
        audio_clean_latents = init_audio_latent
    else:
        audio_clean_latents = torch.zeros_like(audio_noise)
    audio_noise = noise_scale * audio_noise + (1-noise_scale) * audio_clean_latents
    audio_denoising_mask = audio_patchifier.patchify(torch.ones(*audio_latent_shape.mask_shape().to_torch_shape(), device=device, dtype=torch.float32))
    audio_latent_coords = audio_patchifier.get_patch_grid_bounds(output_shape=audio_latent_shape,device=device)

    video_latent_state = LatentState(
        latent=video_noise,
        denoise_mask=video_denoising_mask,
        positions=positions.to(dtype),
        clean_latent=video_clean_latents
    )

    audio_latent_state = LatentState(
        latent=audio_noise,
        denoise_mask=audio_denoising_mask,
        positions=audio_latent_coords,
        clean_latent=audio_clean_latents
    )

    return video_latent_state, audio_latent_state, video_latent_shape, audio_latent_shape


def convert_to_uint8(frames: torch.Tensor) -> torch.Tensor:
    frames = (((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    frames = einops.rearrange(frames[0], "c f h w -> f h w c")
    return frames


def decode_video(video_latent: torch.Tensor, video_decoder: VideoDecoder,  tiling_config: TilingConfig | None = None):
    if tiling_config is not None:
        decoded_video = []
        for frames in video_decoder.tiled_decode(video_latent, tiling_config):
            decoded_video.append(convert_to_uint8(frames))
        decoded_video = torch.cat(decoded_video, dim=0)
    else:
        decoded_video = convert_to_uint8(video_decoder(video_latent))
    
    return decoded_video.cpu().numpy()

def decode_audio(audio_latent: torch.Tensor, audio_decoder: AudioDecoder, vocoder: Vocoder):
    decoded_audio = audio_decoder(audio_latent)
    decoded_audio = vocoder(decoded_audio).squeeze(0).float() # 2 audio_len

    return decoded_audio.cpu().numpy()

def decode_video_audio(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    video_decoder: VideoDecoder,
    audio_decoder: AudioDecoder,
    vocoder: Vocoder,
    tiling_config: TilingConfig | None = None
):
      
    decoded_video = decode_video(video_latent, video_decoder, tiling_config)
    decoded_audio = decode_audio(audio_latent, audio_decoder, vocoder)

    return decoded_video, decoded_audio


def upsample_video(latent: torch.Tensor, video_encoder: VideoEncoder, upsampler: LatentUpsampler) -> torch.Tensor:
    """
    Apply upsampling to the latent representation using the provided upsampler,
    with normalization and un-normalization based on the video encoder's per-channel statistics.
    Args:
        latent: Input latent tensor of shape [B, C, F, H, W].
        video_encoder: VideoEncoder with per_channel_statistics for normalization.
        upsampler: LatentUpsampler module to perform upsampling.
    Returns:
        torch.Tensor: Upsampled and re-normalized latent tensor.
    """
    latent = video_encoder.per_channel_statistics.un_normalize(latent)
    latent = upsampler(latent)
    latent = video_encoder.per_channel_statistics.normalize(latent)
    return latent




def split_state_dict(state_dict):
    dit_state_dict = {}
    video_encoder_state_dict = {}
    video_decoder_state_dict = {}
    audio_encoder_state_dict = {}
    audio_decoder_state_dict = {}
    vocoder_state_dict = {}
    embeddings_processor_state_dict = {}

    for key in state_dict.keys():
        if key.startswith('text_embedding_projection.aggregate_embed.') or key.startswith('text_embedding_projection.video_aggregate_embed.') or \
            key.startswith('text_embedding_projection.audio_aggregate_embed.') or key.startswith('model.diffusion_model.video_embeddings_connector.') or \
            key.startswith('model.diffusion_model.audio_embeddings_connector.'):
            new_key = key.replace('text_embedding_projection.aggregate_embed.', 'feature_extractor.aggregate_embed.').replace('text_embedding_projection.video_aggregate_embed.', 
                    'feature_extractor.video_aggregate_embed.').replace('text_embedding_projection.audio_aggregate_embed.', 
                    'feature_extractor.audio_aggregate_embed.').replace('model.diffusion_model.video_embeddings_connector.', 
                    'video_connector.').replace('model.diffusion_model.audio_embeddings_connector.', 'audio_connector.')
            embeddings_processor_state_dict[new_key] = state_dict[key]
        elif key.startswith('model.diffusion_model.'):
            new_key = key.replace('model.diffusion_model.','')
            dit_state_dict[new_key] = state_dict[key]
        elif key.startswith('vae.encoder.'):
            new_key = key.replace('vae.encoder.', '')
            video_encoder_state_dict[new_key] = state_dict[key]
        elif key.startswith('vae.decoder.'):
            new_key = key.replace('vae.decoder.', '')
            video_decoder_state_dict[new_key] = state_dict[key]
        elif key.startswith('vae.per_channel_statistics.'):
            new_key = key.replace('vae.per_channel_statistics.', 'per_channel_statistics.')
            video_encoder_state_dict[new_key] = state_dict[key]
            video_decoder_state_dict[new_key] = state_dict[key]
        elif key.startswith('audio_vae.encoder.'):
            new_key = key.replace('audio_vae.encoder.', '')
            audio_encoder_state_dict[new_key] = state_dict[key]
        elif key.startswith('audio_vae.decoder.'):
            new_key = key.replace('audio_vae.decoder.', '')
            audio_decoder_state_dict[new_key] = state_dict[key]
        elif key.startswith('audio_vae.per_channel_statistics.'):
            new_key = key.replace('audio_vae.per_channel_statistics.', 'per_channel_statistics.')
            audio_encoder_state_dict[new_key] = state_dict[key]
            audio_decoder_state_dict[new_key] = state_dict[key]
        elif key.startswith('vocoder.'):
            new_key = key.replace('vocoder.', '')
            vocoder_state_dict[new_key] = state_dict[key]

    return dit_state_dict, video_encoder_state_dict, video_decoder_state_dict, audio_encoder_state_dict, audio_decoder_state_dict, vocoder_state_dict, embeddings_processor_state_dict


def load_lora_state_dict(lora_path):
    state_dict = load_file(lora_path)
    state_dict = {k.replace('diffusion_model.', ''):v for k,v in state_dict.items()}
    state_dict_lora = {}
    for key in state_dict.keys():
        if key.startswith('text_embedding_projection'):
            continue

        state_dict_lora[key] = state_dict[key]
    
    return state_dict_lora


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, sampling_rate=16000, filter_length=1024, hop_length=160, win_length=1024, n_mel=64, mel_fmin=0, mel_fmax=8000):
    if torch.min(y) < -1.0:
        print("train min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("train max value is ", torch.max(y))

    mel = librosa_mel_fn(
        sr=sampling_rate,
        n_fft=filter_length,
        n_mels=n_mel,
        fmin=mel_fmin,
        fmax=mel_fmax,
    )
    mel_basis = (torch.from_numpy(mel).float().to(y.device))
    hann_window = torch.hann_window(win_length).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (
            int((filter_length - hop_length) / 2),
            int((filter_length - hop_length) / 2),
        ),
        mode="reflect",
    ).squeeze(1)

    stft_spec = torch.stft(
        y,
        filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    stft_spec = torch.abs(stft_spec)

    mel = spectral_normalize_torch(torch.matmul(mel_basis, stft_spec))

    return mel.permute(0,2,1), stft_spec