import os
import argparse
import random
import numpy as np
import torch
import pandas as pd
import torchaudio
from loguru import logger
from .hunyuanvideo_foley.utils.model_utils import load_model
from .hunyuanvideo_foley.utils.feature_utils import feature_process
from .hunyuanvideo_foley.utils.model_utils import denoise_process
from .hunyuanvideo_foley.utils.media_utils import merge_audio_video

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def set_manual_seed(global_seed):
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

def infer(video_path, prompt, cfg, model_path, config_path, guidance_scale=4.5, num_inference_steps=50):
    device = setup_device("auto")
    
    visual_feats, text_feats, audio_len_in_s = feature_process(
        video_path,
        prompt,
        cfg,
        device,
        model_path
    )

    audio, sample_rate = denoise_process(
        visual_feats,
        text_feats,
        audio_len_in_s,
        model_path,
        config_path,
        device,
        cfg,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )
    return audio[0], sample_rate
    
def generate_audio(model_dict, cfg, csv_path, output_dir, guidance_scale=4.5, num_inference_steps=50):
    os.makedirs(output_dir, exist_ok=True)
    test_df = pd.read_csv(csv_path)

    for index, row in test_df.iterrows():
        video_path = row['video']
        prompt = row['prompt']

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Prompt: {prompt}")

        output_audio_path = os.path.join(output_dir, f"{index:04d}.wav")
        output_video_path = os.path.join(output_dir, f"{index:04d}.mp4")

        if not os.path.exists(output_audio_path) or not os.path.exists(output_video_path):
            audio, sample_rate = infer(video_path, prompt, model_dict, cfg, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
            torchaudio.save(output_audio_path, audio, sample_rate)

            merge_audio_video(output_audio_path, video_path, output_video_path)

    logger.info(f"All audio files saved to {output_dir}")    

def setup_device(device_str, gpu_id=0):
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"Using CUDA device: {device}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        if device_str == "cuda":
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device(device_str)
        logger.info(f"Using specified device: {device}")
    
    return device
    
def process_single_video(video_path, prompt, model_dict, cfg, output_dir, args):
    logger.info(f"Processing single video: {video_path}")
    logger.info(f"Text prompt: {prompt}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_audio_path = os.path.join(output_dir, f"{video_name}_generated.wav")
    output_video_path = os.path.join(output_dir, f"{video_name}_with_audio.mp4")
    
    if args.skip_existing and os.path.exists(output_audio_path):
        logger.info(f"Skipping existing audio file: {output_audio_path}")
        if args.save_video and os.path.exists(output_video_path):
            logger.info(f"Skipping existing video file: {output_video_path}")
            return
    
    audio, sample_rate = infer(
        video_path, prompt, model_dict, cfg, 
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps
    )
    
    torchaudio.save(output_audio_path, audio, sample_rate)
    logger.info(f"Audio saved to: {output_audio_path}")
    
    if args.save_video:
        merge_audio_video(output_audio_path, video_path, output_video_path)
        logger.info(f"Video with audio saved to: {output_video_path}")    
        
class HunyuanVideoFoleyAudioGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("HVFMODEL_CFG",),
                "config_path": ("STRING",),
                "video_path": ("STRING",),
                "prompt": ("STRING", {"multiline":True}),
                "guidance_scale": ("FLOAT",{"default":4.5,"min":0.0,"max":99.9,"step":0.1}),
                "steps": ("INT", {"default":50, "min":1, "max":200, "step":1}),
                "seed": ("INT", {"default":1234,"max":0x7fffffff}),
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("output_audio_path", "output_video_path",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoFoleyWrapper"
    OUTPUT_NODE = True
    
    def process(self, cfg, config_path, video_path, prompt, guidance_scale, steps, seed): 
        set_manual_seed(seed)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        model_path = os.path.join(comfy_path,"models","hunyuanvideo","foley")
        
        output_dir = os.path.join(comfy_path, "output")
        
        output_audio_path = os.path.join(output_dir, f"{video_name}_generated.wav")
        output_video_path = os.path.join(output_dir, f"{video_name}_with_audio.mp4")           
        
        audio, sample_rate = infer(
            video_path, prompt, cfg, 
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            model_path=model_path,
            config_path=config_path
        )
        
        torchaudio.save(output_audio_path, audio, sample_rate)
        logger.info(f"Audio saved to: {output_audio_path}")
        
        merge_audio_video(output_audio_path, video_path, output_video_path)
        logger.info(f"Video with audio saved to: {output_video_path}")           
        
        return (output_audio_path, output_video_path,)
        
class HunyuanVideoFoleyModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config":(["hunyuanvideo-foley-xxl.yaml"],),
            }
        }

    RETURN_TYPES = ("HVFMODEL_CFG","STRING",)
    RETURN_NAMES = ("cfg","config_path",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoFoleyWrapper"
    
    def process(self, config):            
        #device = setup_device("auto")
        
        config_path = os.path.join(script_directory,"configs",config)
        #model_path = os.path.join(comfy_path,"models","hunyuanvideo","foley")
        
        cfg = load_model(config_path)
        
        return (cfg, config_path,)        

NODE_CLASS_MAPPINGS = {
    "HunyuanVideoFoleyAudioGen": HunyuanVideoFoleyAudioGen,
    "HunyuanVideoFoleyModelLoader": HunyuanVideoFoleyModelLoader,
    }
    
NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoFoleyAudioGen": "HunyuanVideo Foley - Audio Generator",
    "HunyuanVideoFoleyModelLoader": "HunyuanVideo Foley - Model Loader",
    }