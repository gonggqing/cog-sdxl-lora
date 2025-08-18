# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Cog-based SDXL image generation model with advanced LoRA support and custom model integration. The project is designed for deployment on Replicate and provides a Flux-compatible interface for easy migration from Flux models.

## Development Commands

### Testing
- `python test_folder/simple_test.py` - Quick functionality test
- `python test_folder/test_local_generation.py` - Complete generation test
- `python test_folder/test_hf_download.py` - Test HuggingFace model downloading

### Deployment
- `cog predict -i prompt="a cat in a hat"` - Test prediction locally with cog
- `cog build` - Build the Docker container
- `cog push <model_name>` - Deploy to Replicate

### Dependencies
- `pip install -r requirements.txt` - Install Python dependencies
- Requirements include PyTorch 2.6.0, diffusers 0.32.0, transformers, and related ML libraries

## Architecture

### Core Components

**Main Predictor** (`predict.py`):
- Implements Cog BasePredictor interface
- Handles Flux-compatible parameter mapping for easy migration
- Supports both standard SDXL and custom models (Illustrious XL by default)
- Multi-LoRA loading and management with intelligent caching

**Weight Management** (`weights.py`):
- `SDXLLoRACache` - LRU cache for LoRA models with memory management
- Supports HuggingFace, CivitAI, and direct URL downloads
- Automatic retry logic and validation of downloaded models
- Environment-aware caching (Docker vs local development)

**Utilities** (`utils.py`):
- Model configuration and custom model support
- LoRA weight loading and merging functions
- Download utilities using pget for parallel downloads
- System info logging and validation helpers

### Model Strategy

**Primary Model**: Illustrious XL UNet is pre-downloaded during Docker build (6.5GB) for instant access. This model is optimized for anime/illustration generation with automatic parameter tuning (CFG=6.0).

**Fallback**: Standard SDXL components are available via `use_standard_sdxl=True` parameter for realistic content generation.

**Model Switching**: Runtime switching between models without downloads, as both are cached during container build.

### LoRA System

**Multi-LoRA Support**:
- Primary LoRA via `lora_weights` parameter
- Secondary LoRA via `extra_lora` parameter for style mixing
- Independent scaling controls (`lora_scale`, `extra_lora_scale`)
- Intelligent caching with LRU eviction to manage memory

**LoRA Sources**:
- HuggingFace models: `huggingface.co/author/model-name`
- CivitAI models: `civitai.com/models/12345`
- Direct URLs to `.safetensors` files
- Replicate models: `author/model-name` format

### Flux Compatibility Interface

The predictor provides Flux-compatible parameters to enable easy migration:
- Similar parameter names and behaviors
- Automatic parameter mapping and validation
- Maintains Flux workflow patterns while leveraging SDXL quality

## Key Implementation Details

**Memory Management**:
- Efficient model caching and cleanup
- LRU-based LoRA cache to prevent memory overflow
- Proper GPU memory management for CUDA and MPS backends

**Performance Optimizations**:
- Pre-cached models during Docker build (no cold start downloads)
- Parallel downloads using pget when needed
- Optimized schedulers (DPM, Euler, DDIM, etc.)

**Safety and Quality**:
- NSFW safety checker with option to disable
- Professional-grade sampling algorithms
- Comprehensive error handling with graceful fallbacks

## Testing Strategy

The `test_folder/` contains three main test scripts:
- `simple_test.py` - Basic generation test bypassing cog parameters
- `test_local_generation.py` - Complete functionality test with all features
- `test_hf_download.py` - HuggingFace integration testing

All tests are designed to work both locally and in the Docker environment with automatic fallbacks.

## Important Notes

- Models are pre-downloaded during Docker build to eliminate cold starts
- The system supports both CUDA and Apple Silicon MPS backends
- Illustrious XL is the default model, optimized for anime/illustration content
- Standard SDXL fallback is available for realistic content via parameter flag
- All LoRA models are cached locally with intelligent memory management