# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import io
import os
import random
import numpy as np
import torch
import torchaudio
import gc
import platform
import subprocess
from omegaconf import OmegaConf

from .node_utils import load_images,nomarl_upscale,is_directory_with_files
from .inference import load_StableAvatar_model,pre_data_process,infer_StableAvatar
from .vocal_seperator import separator_audio
import folder_paths


MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# pre dir
current_path = os.path.dirname(os.path.abspath(__file__))

weigths_current_path = os.path.join(folder_paths.models_dir, "StableAvatar")
if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)

weigths_DIT_current_path = os.path.join(weigths_current_path, "transformer")
if not os.path.exists(weigths_DIT_current_path):
    os.makedirs(weigths_DIT_current_path)

weigths_audio_current_path = os.path.join(weigths_current_path, "wav2vec2-base-960h")
if not os.path.exists(weigths_audio_current_path):
    os.makedirs(weigths_audio_current_path)


# ffmpeg
ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
            print("For example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


# *****************main***************

class StableAvatar_LoadModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": (folder_paths.get_filename_list("diffusion_models"),),
                "vae": (folder_paths.get_filename_list("vae"),),
                "enable_teacache": ("BOOLEAN", {"default": False},),
                "use_mmgp": (["LowRAM_LowVRAM","None", "VerylowRAM_LowVRAM","LowRAM_HighVRAM","HighRAM_LowVRAM","HighRAM_HighVRAM" ],), 
                "GPU_memory_mode": (["model_cpu_offload_and_qfloat8", "model_cpu_offload","None","sequential_cpu_offload", ],), 
                "weight_dtype": (["bfloat16", "float16", "float32"],),
                },
        }
    
    RETURN_TYPES = ("MODEL_PIPE_SA", "MODEL_INFO_SA")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "main_loader"
    CATEGORY = "StableAvatar"
    
    def main_loader(self, transformer,vae,enable_teacache,use_mmgp,GPU_memory_mode,weight_dtype):

        vae_path=folder_paths.get_full_path( "vae",vae)
        transformer_path=folder_paths.get_full_path("diffusion_models",transformer)

        # 复用模型
        echo_audio_files=os.path.join(folder_paths.models_dir,"echo_mimic/wav2vec2-base-960h")
        pretrained_wav2vec_path= echo_audio_files if is_directory_with_files(echo_audio_files) else weigths_audio_current_path

        echo_dit_files=os.path.join(folder_paths.models_dir,"echo_mimic/transformer")
        pretrained_dit_path= echo_dit_files if is_directory_with_files(echo_dit_files) else weigths_DIT_current_path

        config = OmegaConf.load(os.path.join(current_path, "StableAvatar/deepspeed_config/wan2.1/wan_civitai.yaml"))
        args = {"transformer_path": transformer_path,"pretrained_dit_path":pretrained_dit_path,"teacache_offload":True,"sample_steps":25,
                "enable_teacache":enable_teacache,"ulysses_degree":1,"weight_dtype":weight_dtype,"t5_fsdp":True, "t5_cpu":False,"fsdp_dit":True,"ring_degree":1,"num_skip_start_steps":5,
                "pretrained_model_name_or_path":os.path.join(current_path, "Wan2.1-Fun-V1.1-1.3B-InP"),"GPU_memory_mode":GPU_memory_mode,"teacache_threshold":0.10,"local_rank":1,
                "pretrained_wav2vec_path":pretrained_wav2vec_path, "temporal_compression_ratio":4,"input_perturbation":0,
                }
        args=OmegaConf.create(args)

        if weight_dtype == "bfloat16":
            weight_dtype_ = torch.bfloat16
        elif weight_dtype == "float16":
            weight_dtype_ = torch.float16
        elif weight_dtype == "float32":
            weight_dtype_ = torch.float32

        model, tokenizer,temporal_compression_ratio= load_StableAvatar_model(args,vae_path, config, device,weight_dtype_,use_mmgp)
        args.temporal_compression_ratio=temporal_compression_ratio
       
        info={"args":args,"tokenizer":tokenizer,"weight_dtype":weight_dtype_,}
       
        print("##### model  loaded #####")
     
        return (model,info)


class StableAvatar_Predata:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "info": ("MODEL_INFO_SA",),
                "clip": ("CLIP",),
                "clip_vision": ("CLIP_VISION",), 
                "image": ("IMAGE",),  # [B,H,W,C], C=3
                "audio": ("AUDIO",),
                "prompt": ("STRING", {"multiline": True,"default":"A middle-aged woman with short light brown hair, wearing pearl earrings and a blue blazer,"
                " is speaking passionately in front of a blurred background resembling a government building. Her mouth is open mid-phrase, her expression is"
                " engaged and energetic, and the lighting is bright and even, suggesting a television interview or live broadcast. The scene gives the impression she is singing with conviction and purpose."}),
                "negative_prompt" :("STRING", {"multiline": True,"default":"色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" }),
                "width": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 16, "display": "number"}),
                "height": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 16, "display": "number"}),   
                "fps": ("FLOAT", {"default": 25.0, "min": 5.0, "max": 120.0}),
                "duration": ("INT", {"default": 5, "min": 0, "max": 3600, "step": 1, "display": "number"}),
                "audio_separator": ("BOOLEAN", {"default": True},),
                },

        }
    
    RETURN_TYPES = ("MODEL_EMB_E", )
    RETURN_NAMES = ("emb",)
    FUNCTION = "main_loader"
    CATEGORY = "StableAvatar"
    
    def main_loader(self,info,clip,clip_vision, image,audio,prompt,negative_prompt,width,height,fps,duration,audio_separator):

        args=info.get("args")
        weight_dtype=info.get("weight_dtype")
        args.clip_sample_n_frames=81

        # pre img
        infer_img = nomarl_upscale(image, width, height)

        #pre audio
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
        audio_file = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")

        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        if duration > 0:
            max_samples = int(duration * sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        buff = io.BytesIO()
        torchaudio.save(buff, waveform, sample_rate, format="FLAC")
        with open(audio_file, 'wb') as f:
            f.write(buff.getbuffer())
            
        # get vocal if infer a song
        if audio_separator:
            audio_separator_model_file=os.path.join(weigths_current_path,"Kim_Vocal_2.onnx")
            validation_driven_audio_path=separator_audio(audio_file, audio_separator_model_file,folder_paths.get_temp_directory(),audio_file_prefix)
        else:
            validation_driven_audio_path=audio_file
        args.validation_driven_audio_path=validation_driven_audio_path

        # get emb
        emb = pre_data_process(clip,clip_vision,info.get("tokenizer"),prompt,negative_prompt,infer_img,device,width,height,args,weight_dtype)
        emb.update({"audio_file_prefix":audio_file_prefix,"fps":fps,"height":height,"width":width,"enable_teacache": args.enable_teacache,
                    "num_skip_start_steps":args.num_skip_start_steps,
                    "teacache_offload":args.teacache_offload,
                    "teacache_threshold":args.teacache_threshold,
                    "sample_steps":args.sample_steps,
                    "pretrained_model_name_or_path":args.pretrained_model_name_or_path,
                    "weight_dtype":weight_dtype,
                    })

        return (emb,)


class StableAvatar_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_PIPE_SA",),
                "emb": ("MODEL_EMB_E",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "sample_text_guide_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "sample_audio_guide_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "overlap_window_length": ("INT", {"default": 5, "min": 5, "max": 15,"step": 1,}),     
                "save_video": ("BOOLEAN", {"default": False},),
             
                  },    
        }
    
    RETURN_TYPES = ("LATENT",  "FLOAT",)
    RETURN_NAMES = ("latent",  "frame_rate",)
    FUNCTION = "em_main"
    CATEGORY = "StableAvatar"
    
    def em_main(self,model, emb, seed, cfg, sample_text_guide_scale,sample_audio_guide_scale,steps, overlap_window_length,save_video,):

        emb["motion_frame"]=25
        emb["sample_steps"]=steps
        frame_rate = float(emb.get("fps"))
        samples = infer_StableAvatar(model, emb,seed, cfg,device, steps,frame_rate, sample_text_guide_scale,sample_audio_guide_scale, overlap_window_length,save_video,emb.get("weight_dtype",torch.bfloat16))
        #print("samples.shape:",samples.shape)
        gc.collect()
        torch.cuda.empty_cache()
        return ({"samples":samples}, frame_rate,)



NODE_CLASS_MAPPINGS = {
    "StableAvatar_LoadModel": StableAvatar_LoadModel,
    "StableAvatar_Predata":StableAvatar_Predata,
    "StableAvatar_Sampler": StableAvatar_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableAvatar_LoadModel": "StableAvatar_LoadModel",
    "StableAvatar_Predata": "StableAvatar_Predata",
    "StableAvatar_Sampler": "StableAvatar_Sampler",
}
