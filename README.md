# SDXL LoRA Enhanced

Advanced SDXL image generation with multi-LoRA support and custom model integration. Deployed on Replicate with HuggingFace model hosting for maximum reliability.

## 🚀 Features

- **High-Quality SDXL Generation** with professional results
- **Multiple LoRA Support** from HuggingFace, CivitAI, and direct URLs  
- **Custom Model Integration** including specialized models like Illustrious XL
- **HuggingFace Integration** for reliable model hosting and downloads
- **Flux-Compatible Interface** for easy migration and familiarity
- **Advanced Parameter Control** with professional-grade options
- **Robust Download System** with pget and retry logic
- **Apple Silicon & CUDA Support** for maximum compatibility

## 📦 Quick Start

### Replicate Usage (Recommended)
```python
import replicate

# Basic generation
output = replicate.run(
    "drawhisper/illustrious-xl-lora",
    input={
        "prompt": "a beautiful mountain landscape, sunset, high quality",
        "aspect_ratio": "16:9",
        "num_inference_steps": 25
    }
)
```

**🔗 Model URL**: https://replicate.com/drawhisper/illustrious-xl-lora

### Local Development
```bash
# Clone and setup
git clone <repository-url>
cd cog-sdxl-lora
conda create -n cog python=3.11
conda activate cog
pip install -r requirements.txt

# Test locally (uses fallback to local models)
python test_folder/simple_test.py
```

## 🎨 Usage Examples

### Basic Generation (Illustrious XL - Default)
```python
import replicate

# Illustrious XL is loaded by default - optimized for anime/illustration
output = replicate.run("drawhisper/illustrious-xl-lora", input={
    "prompt": "anime girl, colorful hair, detailed eyes, masterpiece",
    "aspect_ratio": "2:3", 
    "num_inference_steps": 28
    # guidance=6.0 and negative prompt are automatically optimized
})
```

### Realistic Content Generation
```python
output = replicate.run("drawhisper/illustrious-xl-lora", input={
    "prompt": "a beautiful mountain landscape, sunset, high quality",
    "use_standard_sdxl": True,  # Switch to standard SDXL for realistic content
    "aspect_ratio": "16:9",
    "guidance": 7.5,
    "num_inference_steps": 25
})
```

### With LoRA
```python
output = replicate.run("drawhisper/illustrious-xl-lora", input={
    "prompt": "portrait in artistic style",
    "lora_weights": "huggingface.co/author/style-lora",
    "lora_scale": 0.8,
    "num_inference_steps": 25
})
```

### Multiple LoRAs
```python
output = replicate.run("drawhisper/illustrious-xl-lora", input={
    "prompt": "fantasy character, detailed artwork", 
    "lora_weights": "huggingface.co/author/character-lora",
    "extra_lora": "huggingface.co/author/style-lora",
    "lora_scale": 0.7,
    "extra_lora_scale": 0.5
})
```

## 🛠️ Parameters

### Core Parameters
- `prompt`: Text description of the desired image
- `num_outputs`: Number of images to generate (1-4)
- `aspect_ratio`: Image dimensions ("1:1", "16:9", "2:3", etc.)
- `num_inference_steps`: Quality vs speed (10-50)
- `guidance`: Prompt adherence (1.0-20.0)

### LoRA Parameters
- `lora_weights`: Primary LoRA model URL
- `lora_scale`: LoRA influence strength (0.0-3.0)
- `extra_lora`: Secondary LoRA for style mixing
- `extra_lora_scale`: Secondary LoRA strength

### Advanced Options
- `base_model`: Custom model selection
- `negative_prompt`: What to avoid in generation
- `seed`: For reproducible results
- `go_fast`: Quick generation mode
- `output_format`: webp, png, or jpg
- `disable_safety_checker`: Skip NSFW filtering

## 🎯 Model Architecture

### Default Model: Illustrious XL
- **Primary**: Illustrious XL UNet loaded by default during setup
- **Optimized for**: High-quality anime/illustration generation
- **Auto-optimization**: CFG=6.0, specialized negative prompts
- **Pre-cached**: Downloaded during Docker build (6.5GB)

### Fallback: Standard SDXL
- **Available via**: `use_standard_sdxl=True` parameter
- **Optimized for**: Realistic content, landscapes, photography
- **Built-in**: Standard SDXL components (VAE, text encoders, UNet)

### Model Integration Strategy
- **Build-time**: Illustrious XL UNet pre-downloaded and ready
- **Runtime**: Instant switching between Illustrious XL and standard SDXL
- **No downloads**: All models cached during container build

### LoRA Sources
- **HuggingFace**: `huggingface.co/author/model-name` (recommended)
- **CivitAI**: `civitai.com/models/12345`
- **Direct URLs**: Any `.safetensors` file URL
- **Replicate**: `author/model-name` format

## 📚 Documentation

- **[Quick Start Guide](documents/QUICK_START.md)** - Get up and running
- **[Deployment Guide](documents/REPLICATE_DEPLOYMENT_GUIDE.md)** - Complete Replicate deployment
- **[Test Folder](test_folder/)** - Local testing scripts

## 🏗️ Architecture

```
cog-sdxl-lora/
├── predict.py              # Main predictor with flux interface
├── weights.py              # LoRA caching and download system  
├── utils.py                # Custom model configs and utilities
├── cog.yaml               # Replicate deployment config
├── requirements.txt       # Python dependencies
├── documents/            # Documentation and guides
└── test_folder/         # Local testing scripts
```

## 🔧 Technical Features

### Multi-LoRA System
- Intelligent caching with LRU eviction
- Simultaneous loading of multiple LoRAs
- Automatic weight merging and scaling
- Memory-efficient model management

### Illustrious XL as Primary Model
- **Default choice**: Illustrious XL optimized for high-quality anime/illustration generation
- **Build-time caching**: Pre-downloaded during Docker build (6.5GB) for instant access
- **Automatic parameter optimization**: Model-specific CFG (6.0) and negative prompts  
- **SDXL fallback**: Standard SDXL available for realistic content generation

### Performance Optimizations
- **Efficient memory management**: Proper cleanup and caching
- **CUDA and MPS backend support**: Works on both NVIDIA and Apple Silicon
- **Model caching**: Downloaded models persist between predictions
- **Network resilience**: Retry logic handles temporary network issues

### Quality Features
- **Professional safety checking**: NSFW content filtering
- **Advanced sampling algorithms**: Multiple scheduler options
- **Flexible output formats**: WebP, PNG, JPEG with quality control
- **Comprehensive error handling**: Graceful fallbacks and detailed error messages

## 🧪 Testing

Run local tests to verify functionality:

```bash
# Quick test
python test_folder/simple_test.py

# Full test suite  
python test_folder/test_local_generation.py
```

Tests cover:
- Basic SDXL generation
- Custom model loading
- LoRA functionality
- Multi-LoRA combinations
- Flux interface compatibility

## 🚀 Performance

- **Generation Time**: 
  - Default SDXL: 20-30 seconds (pre-cached during build)
  - Illustrious XL: 25-35 seconds (pre-cached during build)
  - Cold start: ~45-60 seconds (model loading only, no downloads)
- **Memory Usage**: 6-12GB VRAM depending on model and settings
- **Quality**: Professional-grade SDXL output with custom model enhancements
- **Compatibility**: CUDA, MPS (Apple Silicon), and CPU fallback
- **Network**: Models pre-downloaded during build for maximum reliability

## 📊 Comparison

| Feature | SDXL LoRA Enhanced | Standard SDXL | Flux |
|---------|-------------------|---------------|------|
| Multiple LoRAs | ✅ | ❌ | ✅ |
| Custom Models | ✅ | ❌ | ❌ |
| Flux Interface | ✅ | ❌ | ✅ |
| SDXL Quality | ✅ | ✅ | ❌ |
| Easy Deployment | ✅ | ❌ | ✅ |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Stability AI** for SDXL
- **Replicate** for deployment platform
- **HuggingFace** for model hosting
- **Community** for LoRA models and feedback

---

**Ready to create amazing images with SDXL LoRA Enhanced!** 🎨

For detailed setup instructions, see [REPLICATE_DEPLOYMENT_GUIDE.md](documents/REPLICATE_DEPLOYMENT_GUIDE.md).