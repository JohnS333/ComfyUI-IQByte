@echo off
cd /d "D:\ComfyUI"

:: Set Environment Variables for DirectML
set DML_VISIBLE_DEVICES=0
set PYTORCH_DIRECTML_DEFAULT_DEVICE=0

:: Force Polaris architecture recognition (Optional but recommended for RX 570)
set HSA_OVERRIDE_GFX_VERSION=10.3.0

:: Disable PyTorch compilation to bypass C++ MinGW bugs
set TORCH_COMPILE_DISABLE=1

call .\venv\Scripts\activate

python main.py --directml --use-pytorch-cross-attention --disable-cuda-malloc --highvram --disable-smart-memory --force-fp32

pause