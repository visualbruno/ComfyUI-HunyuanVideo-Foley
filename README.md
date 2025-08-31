# 🌀 ComfyUI Wrapper for [HunyuanVideo-Foley](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley)

## Requires about 11Gb of VRAM ##

![image](https://github.com/visualbruno/ComfyUI-HunyuanVideo-Foley/blob/main/assets/workflow.png)

## 📦 Repository & Models

* **GitHub:** [HunyuanVideo-Foley](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley)
* **Model Weights (HuggingFace):**
  👉 [Main page](https://huggingface.co/tencent/HunyuanVideo-Foley/tree/main)

## 📦 Repository & Models

Make a folder "hunyuanvideo" in the "models" folder in ComfyUI

Go to the folder "hunyuanvideo" and run this command:
```bash
git clone https://huggingface.co/tencent/HunyuanVideo-Foley foley`
```

Directory structure should be like this:

```
ComfyUI/
├── models/
│   ├── hunyuanvideo/
│   │   ├── foley/
|   |   |   └── config.yaml
|   |   |   |   hunyuanvideo_foley.pth
|   |   |   |   synchformer_state_dict.pth
|   |   |   |   vae_128d_48k.pth    
```

## ⚙️ Installation Guide

### 1. Install Python Dependencies

Tested on Windows

#### Install Requirements

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-HunyuanVideo-Foley\win_requirements.txt
```

#### Install audiotools

Go in root folder of ComfyUI

```bash
git clone https://github.com/descriptinc/audiotools`
```

```bash
cd `audiotools`
```

```bash
python_embeded\python.exe -m pip install librosa
```

```bash
python_embeded\python.exe -m pip install .
```



