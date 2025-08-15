import argparse
import gc
import logging
import math
import os
import random
import shutil
import subprocess
from functools import partial

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import librosa
from pathlib import Path
import imageio
import torchvision
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.distributed as dist
import folder_paths
from .StableAvatar.wan.dist import set_multi_gpus_devices
from .StableAvatar.wan.distributed.fsdp import shard_model
from .StableAvatar.wan.models.cache_utils import get_teacache_coefficients
from .StableAvatar.wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from .StableAvatar.wan.models.wan_text_encoder import WanT5EncoderModel
from .StableAvatar.wan.models.wan_vae import AutoencoderKLWan
from .StableAvatar.wan.models.wan_image_encoder import CLIPModel
from .StableAvatar.wan.pipeline.wan_inference_long_pipeline import WanI2VTalkingInferenceLongPipeline

from .StableAvatar.wan.utils.discrete_sampler import DiscreteSampling
from .StableAvatar.wan.utils.fp8_optimization import replace_parameters_by_name, convert_weight_dtype_wrapper, \
    convert_model_weight_to_float8
from .StableAvatar.wan.utils.utils import get_image_to_video_latent, save_videos_grid

logger = get_logger(__name__, log_level="INFO")



def save_video_ffmpeg(gen_video_samples, save_path, vocal_audio_path, fps=25, quality=10):
    def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None, saved_frames_dir=None):
        writer = imageio.get_writer(
            save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
        )
        idx = 0
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            frame_path = os.path.join(saved_frames_dir, f"frame_{idx}.png")
            idx = idx + 1
            imageio.imwrite(frame_path, frame)
            writer.append_data(frame)
        writer.close()

    save_path_tmp = os.path.join(save_path, "video_without_audio.mp4")
    saved_frames_dir = os.path.join(save_path, "animated_images")
    os.makedirs(saved_frames_dir, exist_ok=True)

    # video_audio = (gen_video_samples + 1) / 2  # C T H W
    video_audio = (gen_video_samples / 2 + 0.5).clamp(0, 1)
    video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
    video_audio = np.clip(video_audio * 255, 0, 255).astype(np.uint8)  # to [0, 255]
    save_video(video_audio, save_path_tmp, fps=fps, quality=quality, saved_frames_dir=saved_frames_dir)

    # crop audio according to video length
    _, T, _, _ = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = os.path.join(save_path, "cropped_audio.wav")
    final_command = [
        "ffmpeg",
        "-i",
        vocal_audio_path,
        "-t",
        f'{duration}',
        save_path_crop_audio,
    ]
    subprocess.run(final_command, check=True)


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def get_random_downsample_ratio(sample_size, image_ratio=[],
                                all_choices=False, rng=None):
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list

    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p=number_list_prob)
    else:
        return rng.choice(number_list, p=number_list_prob)


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


# # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.18.0.dev0")




def save_videos_from_pil(pil_images, path, fps=8):
    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if save_fmt == ".mp4":
        with imageio.get_writer(path, fps=fps) as writer:
            for img in pil_images:
                img_array = np.array(img)  # Convert PIL Image to numpy array
                writer.append_data(img_array)

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
            optimize=False,
            lossless=True
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid_png_and_mp4(videos: torch.Tensor, rescale=False, n_rows=6, save_frames_path=None):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)
        outputs.append(x)

    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in outputs]
    num_frames = len(pil_frames)
    for i in range(num_frames):
        pil_frame = pil_frames[i]
        save_path = os.path.join(save_frames_path, f'frame_{i}.png')
        pil_frame.save(save_path)


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)



def load_StableAvatar_model(args,vae_path,config,device,weight_dtype,use_mmgp):
    #args = parse_args()
    
    # device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    sampler_name = "Flow"
    # GPU_memory_mode = "model_full_load"
   
    # fsdp_dit = False
    # weight_dtype=args.weight_dtype
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')), )
    # text_encoder = WanT5EncoderModel.from_pretrained(
    #     os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    #     additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    #     low_cpu_mem_usage=True,
    #     torch_dtype=weight_dtype,
    # )
    # text_encoder = text_encoder.eval()
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    )
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.pretrained_wav2vec_path)
    wav2vec = Wav2Vec2Model.from_pretrained(args.pretrained_wav2vec_path).to("cpu")

    #clip_image_encoder = CLIPModel.from_pretrained(os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')), )
    #clip_image_encoder = clip_image_encoder.eval()

    transformer3d = WanTransformer3DFantasyModel.from_pretrained(
        args.pretrained_dit_path,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
    )
    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        state_dict = torch.load(args.transformer_path, map_location="cpu", weights_only=False)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
    }[sampler_name]

    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    pipeline = WanI2VTalkingInferenceLongPipeline(
        tokenizer=tokenizer,
        #text_encoder=text_encoder,
        vae=vae,
        transformer=transformer3d,
        #clip_image_encoder=clip_image_encoder,
        scheduler=scheduler,
        wav2vec_processor=wav2vec_processor,
        wav2vec=wav2vec,
    )
    # if args.ulysses_degree > 1 or args.ring_degree > 1:
    #     transformer3d.enable_multi_gpus_inference()
    #     if args.fsdp_dit:
    #         shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
    #         pipeline.transformer = shard_fn(pipeline.transformer)
    if use_mmgp!="None":
        from mmgp import offload, profile_type
        pipeline.to("cpu")
        if use_mmgp=="VerylowRAM_LowVRAM":
            offload.profile(pipeline, profile_type.VerylowRAM_LowVRAM)
        elif use_mmgp=="LowRAM_LowVRAM":  
            offload.profile(pipeline, profile_type.LowRAM_LowVRAM)
        elif use_mmgp=="LowRAM_HighVRAM":
            offload.profile(pipeline, profile_type.LowRAM_HighVRAM)
        elif use_mmgp=="HighRAM_LowVRAM":
            offload.profile(pipeline, profile_type.HighRAM_LowVRAM)
        elif use_mmgp=="HighRAM_HighVRAM":
            offload.profile(pipeline, profile_type.HighRAM_HighVRAM)
    elif args.GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer3d, ["modulation", ], device=device)
        transformer3d.freqs = transformer3d.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer3d, exclude_module_name=["modulation", ])
        convert_weight_dtype_wrapper(transformer3d, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

   
    temporal_compression_ratio=vae.config.temporal_compression_ratio
    return  pipeline,tokenizer,temporal_compression_ratio

def pre_data_process(text_encoder,clip_image_encoder,tokenizer,prompt,negative_prompt,infer_img,device,width,height,args,weight_dtype):

    
    clip_sample_n_frames = args.clip_sample_n_frames
    temporal_compression_ratio=args.temporal_compression_ratio
    do_classifier_free_guidance=True
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds=encode_prompt(text_encoder,tokenizer,prompt,negative_prompt,do_classifier_free_guidance,1,device=device,dtype=weight_dtype)

        video_length = int((clip_sample_n_frames - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1 if clip_sample_n_frames != 1 else 1
        input_video, input_video_mask, _ = get_image_to_video_latent([infer_img], None, video_length=video_length, sample_size=[width, height]) # 首尾帧的处理流程
        sr = 16000
        vocal_input, sample_rate = librosa.load(args.validation_driven_audio_path, sr=sr)

        #clip_image = cond_image = Image.open(cond_file_path).convert('RGB')
        clip_image = cond_image = infer_img
        cond_image = cond_image.resize([width, height])
        clip_image = clip_image.resize([width, height])
        clip_image = torch.from_numpy(np.array(clip_image)).permute(2, 0, 1)
        clip_image = clip_image / 255
        clip_image = (clip_image - 0.5) * 2  # C H W
        cond_image = torch.from_numpy(np.array(cond_image)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
        cond_image = cond_image / 255
        cond_image = (cond_image - 0.5) * 2  # normalization
        cond_image = cond_image.to(device)  # 1 C 1 H W

        clip_image = clip_image.to(device, weight_dtype)
        # clip_context = clip_image_encoder([clip_image[:, None, :, :]])
        #print("clip_image:",clip_image.shape) #clip_image: torch.Size([3, 720, 480])
        clip_image=clip_image.permute(1, 2, 0).unsqueeze(0) #comfy need [B,C,H,W] -->[B,H,W,C]
        clip_dict=clip_image_encoder.encode_image(clip_image)
        clip_context =clip_dict["penultimate_hidden_states"].to(device, weight_dtype)
        clip_context = (torch.cat([clip_context, clip_context, clip_context], dim=0) if do_classifier_free_guidance else clip_context)
        clip_image_encoder.patcher.cleanup()

    gc.collect()
    emb={"prompt_embeds":prompt_embeds,"negative_prompt_embeds":negative_prompt_embeds,"vocal_input":vocal_input,"video_length":video_length,
         "clip_image_tensor":cond_image,"clip_context":clip_context,
         "sample_rate":sample_rate,"input_video":input_video,"input_video_mask":input_video_mask,"sr":sr,
         }
    return emb


def infer_StableAvatar(pipeline,args,seed,cfg,device,steps,frame_rate,sample_text_guide_scale,sample_audio_guide_scale,overlap_window_length,save_video,weight_dtype):

    coefficients = get_teacache_coefficients(args.get("pretrained_model_name_or_path")) if args.get("enable_teacache") else None
    if coefficients is not None:
        #print(f"Enable TeaCache with threshold {args.teacache_threshold} and skip the first {args.get(num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients,
            args.get("sample_steps"),
            args.get("teacache_threshold"),
            num_skip_start_steps=args.get("num_skip_start_steps"),
            offload=args.get("teacache_offload")
        )
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        sample = pipeline(
            None,
            num_frames=args.get("video_length"),
            negative_prompt=None,
            height=args.get("height"),
            width=args.get("width"),
            guidance_scale=cfg,
            generator=generator,
            num_inference_steps=steps,
            video=args.get("input_video"),
            mask_video=args.get("input_video_mask"),
            prompt_embeds=args.get("prompt_embeds"),
            negative_prompt_embeds=args.get("negative_prompt_embeds"),
            clip_image=None,
            text_guide_scale=sample_text_guide_scale,
            audio_guide_scale=sample_audio_guide_scale,
            vocal_input_values=args.get("vocal_input"),
            motion_frame=args.get("motion_frame"),
            fps=frame_rate,
            sr=args.get("sr"),
            cond_file_path=None,
            seed=seed,
            overlap_window_length=overlap_window_length,
            clip_image_tensor=args.get("clip_image_tensor"),
            clip_context=args.get("clip_context"),
            weight_dtype=weight_dtype,
        ).videos

        del pipeline
        # audio_file_prefix=args.get("audio_file_prefix")
        # video_path = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_video_without_audio.mp4")
        # outputs=save_videos_grid(sample, video_path,save_video, fps=frame_rate)
    return sample

def encode_prompt(
        text_encoder,
        tokenizer,
        prompt,
        negative_prompt = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        max_sequence_length: int = 512,
        device= None,
        dtype= None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device 

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # prompt_embeds = self._get_t5_prompt_embeds(
            #     prompt=prompt,
            #     num_videos_per_prompt=num_videos_per_prompt,
            #     max_sequence_length=max_sequence_length,
            #     device=device,
            #     dtype=dtype,
            # )
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
       
            prompt_attention_mask = text_inputs.attention_mask

            prompt_embeds=cf_clip(prompt, text_encoder,prompt_attention_mask,device, dtype)[0]

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            text_inputs = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
       
            prompt_attention_mask = text_inputs.attention_mask
            # negative_prompt_embeds = self._get_t5_prompt_embeds(
            #     prompt=negative_prompt,
            #     num_videos_per_prompt=num_videos_per_prompt,
            #     max_sequence_length=max_sequence_length,
            #     device=device,
            #     dtype=dtype,
            # )
            negative_prompt_embeds=cf_clip(negative_prompt, text_encoder,prompt_attention_mask,device, dtype)[0]
       
        return prompt_embeds, negative_prompt_embeds

def cf_clip(txt_list, clip,prompt_attention_mask,device, dtype):
    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
    pos_cond_list = []
    for i in txt_list:
        tokens_p = clip.tokenize(i)
        output_p = clip.encode_from_tokens(tokens_p, return_dict=True)  # {"pooled_output":tensor}
        cond_p = output_p.pop("cond").to(device, dtype)
        #print(cond_p.shape) #torch.Size([1, 231, 768])
        positive=[u[:v] for u, v in zip(cond_p, seq_lens)]
        pos_cond_list.append(positive)
   
    return pos_cond_list


