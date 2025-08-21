# ComfyUI_StableAvatar
[StableAvatar](https://github.com/Francis-Rings/StableAvatar/tree/main): Infinite-Length Audio-Driven Avatar Video Generation,you can try it in ComfyUI


# UPDATE
*  修复TCD和LCM因为全零pad引发的伪影，LCM因为融合机制的问题，批次过渡会有一定的闪烁，但好过之前，cfg不要设置为1。
*  设置GPU_memory_mode 为None 时，关闭mmgp的fp8量化/ when set GPU_memory_mode to None， will disabled mmgp ‘s  FP8 quantize;  
*  when step=4 and lightX2V lora will run in LCM mode/ 步数为4时（搭配lightX2V lora）自动开启lcm； 
*  可以使用lightX2V lora进行10步推理，节省2.5倍的时间/ use lightX2V lora infer in 10 steps.
*  同步官方最新平滑及对数滑动窗口机制代码； 
*  Infinite-Length Audio-Driven / 特点，无限长（音频多长就推理多长） 
*  如果也使用了echomimic V3，会自动调用v3的共用模型 

# 1. Installation

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_StableAvatar.git
```
---
  
# 2. Requirements  
audio-separator仅在推理歌曲时有用。
```
pip install audio-separator --no-deps # optional if need vocal 
pip install -r requirements.txt
```


# 3.models 
3.1 from[FrancisRing/StableAvatar](https://huggingface.co/FrancisRing/StableAvatar) downlaod "Wan2.1_VAE.pth" ,"diffusion_pytorch_model.safetensors" and "config.json ","Kim_Vocal_2.onnx" ,"transformer3d-rec-vec.pt" or "transformer3d-square.pt " 底模有2个可选  
3.2 use comfyui ,[clipvison-h](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/clip_vision) and [umt5_xxl_fp8_e4m3fn_scaled.safetensors ](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders)     
3.3 [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h/tree/main)      
3.4 if use echomimic v3，just only download"transformer3d-rec-vec.pt" or "transformer3d-square.pt " and "Kim_Vocal_2.onnx" / 如果也用echomimic v3，仅需下载底模和Kim_Vocal_2.onnx，会自动调用echomimic的模型   
3.5 可选/optional lora  [kijai](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Lightx2v)
```
├── ComfyUI/models/StableAvatar/transformer 
|         ├── diffusion_pytorch_model.safetensors  # Wan2.1-Fun-V1.1-1.3B-InP transformer #3.13G 务必注意模型同名。
|         ├── config.json
├── ComfyUI/models/StableAvatar/wav2vec2-base-960h
|         ├── all config json files 
|         ├──  model.safetensors
├── ComfyUI/models/clip
|         ├── umt5_xxl_fp8_e4m3fn_scaled.safetensors #comfy
├── ComfyUI/models/clip_vision
|         ├──clipvison-h # 1.26G comfy
├── ComfyUI/models/diffusion_models/
|         ├──transformer3d-rec-vec.pt  # FrancisRing/StableAvatar 二选一
|         ├──transformer3d-square.pt   # FrancisRing/StableAvatar
├── ComfyUI/models/vae
|         ├── Wan2.1_VAE.pth
├── ComfyUI/models/StableAvatar/  # 音频分离用
|         ├──Kim_Vocal_2.onnx
├── ComfyUI/models/loras/    
|         ├──lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors  #KJ
```

# 4.Tips
* 480x832 or 832x480 or 512x512 resolution /模型在这三种分辨率训练，推荐使用
* overlap_window_length 越大越好越慢/ high will get best quality but more times
* step  25~50  如果使用lightX2V 的lora step=10;  
* 二种gpu卸载方式,推荐用mmgp  
* duration>0 时裁切，裁切数值为秒/ if duration>0 will cut the  input audio  
* 推理歌曲时，须开启audio_separator分离人声 / when infer a song  need turn on the audio_separator to get vocal.  
* 暂时不要开启teacache，会花  


# 5. Example
![](https://github.com/smthemex/ComfyUI_StableAvatar/blob/main/example_workflows/example_lora.png)


# 6 .Citation
```
@article{tu2025stableavatar,
  title={StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation},
  author={Tu, Shuyuan and Pan, Yueming and Huang, Yinming and Han, Xintong and Xing, Zhen and Dai, Qi and Luo, Chong and Wu, Zuxuan and Jiang Yu-Gang},
  journal={arXiv preprint arXiv:2508.08248},
  year={2025}
}
```
