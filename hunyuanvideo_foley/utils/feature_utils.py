"""Feature extraction utilities for video and text processing."""

import os
import numpy as np
import torch
import av
import gc
from PIL import Image
from einops import rearrange
from typing import Any, Dict, List, Union, Tuple
from loguru import logger

from .config_utils import AttributeDict
from ..constants import FPS_VISUAL, MAX_VIDEO_DURATION_SECONDS

from .model_utils import load_syncformer_preprocess,load_syncformer_model,load_siglip2_preprocess,load_siglip2_model,load_clap_tokenizer,load_clap_model,load_foley_model,load_dac_model


class FeatureExtractionError(Exception):
    """Exception raised for feature extraction errors."""
    pass

def get_frames_av(
    video_path: str,
    fps: float,
    max_length: float = None,
) -> Tuple[np.ndarray, float]:
    end_sec = max_length if max_length is not None else 15
    next_frame_time_for_each_fps = 0.0
    time_delta_for_each_fps = 1 / fps

    all_frames = []
    output_frames = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        ori_fps = stream.guessed_rate
        stream.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_time = frame.time
                if frame_time < 0:
                    continue
                if frame_time > end_sec:
                    break

                frame_np = None

                this_time = frame_time
                while this_time >= next_frame_time_for_each_fps:
                    if frame_np is None:
                        frame_np = frame.to_ndarray(format="rgb24")

                    output_frames.append(frame_np)
                    next_frame_time_for_each_fps += time_delta_for_each_fps

    output_frames = np.stack(output_frames)

    vid_len_in_s = len(output_frames) / fps
    if max_length is not None and len(output_frames) > int(max_length * fps):
        output_frames = output_frames[: int(max_length * fps)]
        vid_len_in_s = max_length

    return output_frames, vid_len_in_s

@torch.inference_mode()
def encode_video_with_siglip2(x: torch.Tensor, device, batch_size: int = -1):    
    siglip2_model = load_siglip2_model(device)
    
    b, t, c, h, w = x.shape
    if batch_size < 0:
        batch_size = b * t
    x = rearrange(x, "b t c h w -> (b t) c h w")
    outputs = []
    for i in range(0, b * t, batch_size):
        outputs.append(siglip2_model.get_image_features(pixel_values=x[i : i + batch_size]))
    res = torch.cat(outputs, dim=0)
    res = rearrange(res, "(b t) d -> b t d", b=b)
    
    del siglip2_model
    torch.cuda.empty_cache()
    gc.collect()      
    
    return res

@torch.inference_mode()
def encode_video_with_sync(x: torch.Tensor, model_path, device, batch_size: int = -1):
    """
    The input video of x is best to be in fps of 24 of greater than 24.
    Input:
        x: tensor in shape of [B, T, C, H, W]
        batch_size: the batch_size for synchformer inference
    """
    
    syncformer_model = load_syncformer_model(model_path,device)
    
    b, t, c, h, w = x.shape
    assert c == 3 and h == 224 and w == 224

    segment_size = 16
    step_size = 8
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size : i * step_size + segment_size])
    x = torch.stack(segments, dim=1).cuda()  # (B, num_segments, segment_size, 3, 224, 224)

    outputs = []
    if batch_size < 0:
        batch_size = b * num_segments
    x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
    for i in range(0, b * num_segments, batch_size):
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.half):
            outputs.append(syncformer_model(x[i : i + batch_size]))
    x = torch.cat(outputs, dim=0)  # [b * num_segments, 1, 8, 768]
    x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b)
    
    del syncformer_model
    torch.cuda.empty_cache()
    gc.collect()      
    
    return x


@torch.inference_mode()
def encode_video_features(video_path, model_path, device):
    visual_features = {}
    # siglip2 visual features
    
    siglip2_preprocess = load_siglip2_preprocess()
    
    frames, ori_vid_len_in_s = get_frames_av(video_path, FPS_VISUAL["siglip2"])
    images = [Image.fromarray(frame).convert('RGB') for frame in frames]
    images = [siglip2_preprocess(image) for image in images]  # [T, C, H, W]
    
    del siglip2_preprocess
    torch.cuda.empty_cache()
    gc.collect()      
    
    clip_frames = torch.stack(images).to(device).unsqueeze(0)
    #clip_frames = torch.stack(images).unsqueeze(0)
    visual_features['siglip2_feat'] = encode_video_with_siglip2(clip_frames, device).to(device)

    # synchformer visual features
    frames, ori_vid_len_in_s = get_frames_av(video_path, FPS_VISUAL["synchformer"])
    images = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
    
    syncformer_preprocess = load_syncformer_preprocess()
    
    sync_frames = syncformer_preprocess(images).unsqueeze(0)  # [1, T, 3, 224, 224]
    
    del syncformer_preprocess
    torch.cuda.empty_cache()
    gc.collect()      
    
    # [1, num_segments * 8, channel_dim], e.g. [1, 240, 768] for 10s video
    visual_features['syncformer_feat'] = encode_video_with_sync(sync_frames, model_path, device)

    vid_len_in_s = sync_frames.shape[1] / FPS_VISUAL["synchformer"]
    visual_features = AttributeDict(visual_features)

    return visual_features, vid_len_in_s

@torch.inference_mode()
def encode_text_feat(text: List[str], device):
    # x: (B, L)
    
    clap_tokenizer = load_clap_tokenizer()            
    inputs = clap_tokenizer(text, padding=True, return_tensors="pt").to(device)
    
    del clap_tokenizer
    torch.cuda.empty_cache()
    gc.collect()      
    
    clap_model = load_clap_model(device)
    
    outputs = clap_model(**inputs, output_hidden_states=True, return_dict=True)
    
    del clap_model
    torch.cuda.empty_cache()
    gc.collect()      
    
    return outputs.last_hidden_state, outputs.attentions


def feature_process(video_path, prompt, cfg, device, model_path):
    visual_feats, audio_len_in_s = encode_video_features(video_path, model_path, device)
    neg_prompt = "noisy, harsh"
    prompts = [neg_prompt, prompt]
    text_feat_res, text_feat_mask = encode_text_feat(prompts, device)

    text_feat = text_feat_res[1:]
    uncond_text_feat = text_feat_res[:1]

    if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
        text_seq_length = cfg.model_config.model_kwargs.text_length
        text_feat = text_feat[:, :text_seq_length]
        uncond_text_feat = uncond_text_feat[:, :text_seq_length]

    text_feats = AttributeDict({
        'text_feat': text_feat,
        'uncond_text_feat': uncond_text_feat,
    })

    return visual_feats, text_feats, audio_len_in_s
