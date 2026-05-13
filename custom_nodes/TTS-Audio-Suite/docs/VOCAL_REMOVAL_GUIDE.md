# 🤐 Vocal/Noise Removal Guide

Complete guide to using the Vocal/Noise Removal node for audio separation and processing.

## Overview

The **🤐 Noise or Vocal Removal** node is a powerful audio processing tool that can:
- **Separate vocals from instrumentals** (karaoke creation)
- **Remove noise** from audio recordings
- **Remove echo and reverb** artifacts
- **Isolate specific audio stems** (drums, bass, etc.)

## Model Categories & Recommendations

### 🏆 BEST MODELS (2024-2025)

**Top Tier - State of the Art:**
- ★★★ **MELBAND/denoise_mel_band_roformer_sdr_27.99.ckpt** - BEST DENOISING (27.99 dB SDR!) 
- ★★★ **MELBAND/MelBandRoformer_fp32.safetensors** - Kijai's high-quality vocal separation
- ★★★ **model_bs_roformer_ep_317_sdr_12.9755.ckpt** - Reliable transformer-based (12.97 dB SDR)
- ★★☆ **MDX23C-8KFFT-InstVoc_HQ.ckpt** - Minimal artifacts (Karafan architecture)
- ★★☆ **UVR-MDX-NET-vocal_FT.onnx** - Professional vocal extraction (MDX architecture)

### 📂 VR MODELS (Convolutional Neural Networks)

- **UVR-DeEcho-DeReverb.pth** - Post-processing: removes echo/reverb artifacts
- **HP5-vocals+instrumentals.pth** - Balanced vocal/instrumental separation
- **5_HP-Karaoke-UVR.pth** - Optimized for karaoke creation (aggressive vocal removal)
- **6_HP-Karaoke-UVR.pth** - Alternative karaoke model (different tuning)
- **model_bs_roformer_ep_317_sdr_12.9755.ckpt** - ⭐ TRANSFORMER ARCHITECTURE (BEST)
- **UVR-BVE-4B_SN-44100-1.pth** - 4-band processing with spectral normalization
- **UVR-DeNoise.pth** - Noise reduction specialist

### 📂 MDX MODELS (Multi-Dimensional eXtraction)

- **UVR-MDX-NET-vocal_FT.onnx** - Fine-tuned vocal extraction, full-band processing

### 📂 KARAFAN MODELS (Advanced Hybrid)

- **MDX23C-8KFFT-InstVoc_HQ.ckpt** - ⭐ HIGHEST QUALITY (8K FFT, minimal artifacts)

### 🎯 MELBAND ROFORMER MODELS (State-of-the-Art 2025)

**Kijai's Models (Vocal Separation):**
- **MELBAND/MelBandRoformer_fp16.safetensors** - Fast processing (456MB, FP16 precision)
- **MELBAND/MelBandRoformer_fp32.safetensors** - Maximum quality (913MB, FP32 precision)

**ZFTurbo's Models (Denoising Specialists):**
- **MELBAND/denoise_mel_band_roformer_sdr_27.99.ckpt** - ⭐ BEST DENOISING (27.99 dB SDR!)
- **MELBAND/denoise_mel_band_roformer_aggressive_sdr_27.97.ckpt** - Aggressive denoising

**Note:** MelBand models use advanced transformer architecture with rotary embeddings for superior separation quality

## Use Case Workflows

### 🎤 Karaoke Creation (Vocal Removal)
1. **Primary:** `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (aggressive: 12-15)
2. **Post-process:** `UVR-DeEcho-DeReverb.pth` (remove artifacts)
3. **Use:** "remaining" output (clean instrumentals)

### 🎵 Clean Vocal Extraction
1. **Option A:** `UVR-MDX-NET-vocal_FT.onnx` (balanced: 8-12)
2. **Option B:** `MDX23C-8KFFT-InstVoc_HQ.ckpt` (highest quality)
3. **Use:** "extracted voice/noise/echo" output (isolated vocals)

### 🔧 Audio Denoising
1. **Best Quality:** `MELBAND/denoise_mel_band_roformer_sdr_27.99.ckpt` (no aggressiveness setting)
2. **Alternative:** `UVR-DeNoise.pth` (gentle: 5-8)
3. **Use:** "remaining" output (clean audio)
4. **Note:** "extracted" contains removed noise

### 💼 Professional Workflow
1. **Separate:** `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
2. **Clean vocals:** `UVR-DeEcho-DeReverb.pth`
3. **Denoise:** `UVR-DeNoise.pth`
4. **Combine:** Use Merge Audio node

### 🏠 Beginner-Friendly
- **Start with:** `HP5-vocals+instrumentals.pth` (moderate: 8-10)
- **Simple setup:** No post-processing needed

## Model Architectures Explained

### ⚡ ARCHITECTURE DIFFERENCES

- **VR (Vocal Remover):** Fast, magnitude-only processing, good for basic separation
- **MDX (Multi-Dimensional eXtraction):** Hybrid spectrogram/waveform, better quality preservation  
- **Karafan:** Multi-stage ensemble, state-of-the-art quality (highest CPU usage)
- **RoFormer:** Transformer with rotary embeddings, excellent quality
- **MelBand RoFormer:** Advanced transformer with Mel-frequency bands, current SOTA (27+ dB SDR)

## Aggressiveness Settings (0-20)

### 📊 RECOMMENDED VALUES
- **0-5:** Gentle separation, preserves original audio quality
- **6-10:** ⭐ BALANCED (Default: 10) - Good separation with minimal artifacts
- **11-15:** Aggressive separation, may introduce artifacts but better isolation
- **16-20:** Maximum aggression, highest separation but potential quality loss

### 🎯 USE CASES BY AGGRESSIVENESS
- **🎤 Karaoke Creation:** 12-15 (more aggressive)
- **🎵 Vocal Extraction:** 8-12 (balanced)
- **🎼 Preserve Music Quality:** 5-8 (gentle)
- **🔧 Problem Audio:** 15-20 (maximum effort)

## Special Model Types

### ⚠️ MODELS WITH DIFFERENT BEHAVIOR

#### UVR-DeNoise.pth - NOISE REDUCTION (not vocal separation)
- **"extracted voice/noise/echo" output** = extracted noise/artifacts ❌
- **"remaining" output** = cleaned, denoised audio ✅
- **Use case:** Remove background noise, hiss, static

#### UVR-DeEcho-DeReverb.pth - ECHO/REVERB REMOVAL
- **"extracted voice/noise/echo" output** = extracted echo/reverb ❌  
- **"remaining" output** = dry, processed audio ✅
- **Use case:** Clean up live recordings, remove room acoustics

#### HP-Karaoke Models - INVERTED OUTPUTS
- **5_HP-Karaoke-UVR.pth, 6_HP-Karaoke-UVR.pth**
- **Automatically corrected:** System detects and swaps outputs
- **Optimized for:** Aggressive vocal removal for karaoke

## Output Format Options

### 🎵 AUDIO FORMAT SELECTION

**Quality Ranking:**
- **📀 FLAC:** ⭐ BEST - Lossless compression, perfect quality, larger files
- **🎵 WAV:** Uncompressed, perfect quality, largest files  
- **🎧 MP3:** Lossy compression, smaller files, slight quality loss

**Professional Use:** FLAC (default) | **Fast Workflow:** MP3 | **Maximum Quality:** WAV

**File Size Comparison (typical 4-minute song):**
- WAV: ~40MB per stem | FLAC: ~20MB per stem | MP3: ~4MB per stem

## Advanced Techniques

### 🚀 MODEL NOTES
- **MelBand RoFormer** - ✅ NOW SUPPORTED! Best-in-class denoising (27+ dB SDR)
- **SCNet models** - ⚠️ Architecture issues causing audio artifacts
- **MDX23C models** - ⚠️ Tensor alignment errors in current implementation
- **Note:** Aggressiveness setting only affects VR architecture models, not MelBand/MDX/Karafan

### 💡 PRO TIPS
- **Use ensemble combinations** for best results
- **Apply post-processing models** (DeEcho, DeNoise) after primary separation
- **RoFormer models** are current state-of-the-art (2024-2025)
- **Higher quality models** require more processing time and memory
- **For noise reduction models,** use the "remaining" output for your clean audio
- **Chain multiple models** for professional results

### 🔄 CACHING SYSTEM
- **✅ ON (Recommended):** Saves results to disk, dramatically speeds up repeated processing
- **❌ OFF:** Always processes from scratch, ensures fresh results
- **Cache includes:** Model, aggression, format, and audio content in hash
- **Auto-invalidates:** When any parameter changes

## Troubleshooting

### Common Issues
- **Empty/silent output:** Check if using correct output (remaining vs extracted)
- **Poor separation:** Try higher aggressiveness or different model
- **Artifacts:** Use post-processing models or lower aggressiveness
- **Slow processing:** Enable caching, ensure GPU acceleration

### GPU Acceleration
- **Required:** `pip install onnxruntime-gpu` (instead of onnxruntime)
- **Benefits:** 3-10x faster processing for ONNX models
- **Check:** Look for "ONNX Runtime CUDA acceleration enabled" message

---

*For more information, see the main [README](../README.md) and [Project Index](../PROJECT_INDEX.md).*