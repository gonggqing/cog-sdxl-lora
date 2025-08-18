"""
Utility functions for SDXL LoRA Enhanced predictor.
Contains all necessary helper functions to make the project self-contained.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Conditional imports for ML libraries (only needed when actually running predictions)
try:
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0
    from safetensors.torch import load_file
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False


def download_weights(url: str, dest: str):
    """
    Download weights from a URL using pget (parallel downloader).
    Adapted from the original flux predictor to be self-contained.
    
    Args:
        url: URL to download from
        dest: Destination path to extract/save to
    """
    start = time.time()
    print(f"Downloading weights from URL: {url}")
    print(f"Downloading to destination: {dest}")
    
    try:
        subprocess.check_call(["pget", "-x", url, dest])
        print(f"Download completed in {time.time() - start:.2f} seconds")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download weights from {url}: {e}")
    except FileNotFoundError:
        raise RuntimeError(
            "pget command not found. Please ensure pget is installed. "
            "Install with: curl -o /usr/local/bin/pget -L 'https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64' && chmod +x /usr/local/bin/pget"
        )


class TokenEmbeddingsHandler:
    """
    Token embeddings handler for managing custom tokens in text encoders.
    Copied from dataset_and_utils.py to make the project self-contained.
    """
    
    def __init__(self, text_encoders: List, tokenizers: List):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.token_map = {}
    
    def load_embeddings(self, embeddings_path: str):
        """Load embeddings from a .pti file if it exists."""
        if not os.path.exists(embeddings_path):
            print(f"No embeddings file found at {embeddings_path}, skipping.")
            return
        
        try:
            # Load embeddings (implementation would depend on the .pti format)
            print(f"Loading embeddings from {embeddings_path}")
            # This is a placeholder - actual implementation would load the embeddings
            # and update the text encoders accordingly
        except Exception as e:
            print(f"Failed to load embeddings from {embeddings_path}: {e}")


def load_lora_weights_to_unet(unet, lora_path: Path, device: str = "cuda") -> Dict[str, Any]:
    """
    Load LoRA weights into a UNet model.
    
    Args:
        unet: The UNet model to load LoRA into
        lora_path: Path to the LoRA .safetensors file
        device: Device to load the LoRA on
    
    Returns:
        Dictionary containing LoRA tensors for reference
    """
    if not ML_LIBRARIES_AVAILABLE:
        raise RuntimeError("ML libraries (diffusers, safetensors) not available. Install requirements.txt dependencies.")
    
    print(f"Loading LoRA weights from {lora_path}")
    
    # Load LoRA tensors
    tensors = load_file(lora_path)
    
    # Extract rank information from LoRA tensors
    unet_lora_attn_procs = {}
    name_rank_map = {}
    
    # First pass: determine ranks
    for tensor_key, tensor_value in tensors.items():
        if tensor_key.endswith("up.weight"):
            proc_name = ".".join(tensor_key.split(".")[:-3])
            rank = tensor_value.shape[1]
            name_rank_map[proc_name] = rank
    
    # Second pass: create LoRA attention processors
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        
        # Determine hidden size based on block type
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            # Skip if we can't determine the block type
            continue
        
        # Only create LoRA processor if we have rank information
        if name in name_rank_map:
            lora_processor = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=name_rank_map[name],
            )
            unet_lora_attn_procs[name] = lora_processor.to(device)
    
    # Set attention processors and load state dict
    unet.set_attn_processor(unet_lora_attn_procs)
    unet.load_state_dict(tensors, strict=False)
    
    print(f"Successfully loaded LoRA with {len(name_rank_map)} attention processors")
    return tensors


def merge_lora_weights(tensors_list: List[Dict[str, torch.Tensor]], scales: List[float]) -> Dict[str, torch.Tensor]:
    """
    Merge multiple LoRA weight tensors with different scales.
    
    Args:
        tensors_list: List of LoRA tensor dictionaries
        scales: List of scaling factors for each LoRA
    
    Returns:
        Dictionary of merged tensors
    """
    if not tensors_list:
        return {}
    
    if len(tensors_list) == 1:
        # If only one LoRA, just scale it
        merged = {}
        for key, tensor in tensors_list[0].items():
            merged[key] = tensor * scales[0]
        return merged
    
    # Merge multiple LoRAs
    merged = {}
    all_keys = set()
    for tensors in tensors_list:
        all_keys.update(tensors.keys())
    
    for key in all_keys:
        merged_tensor = None
        for i, tensors in enumerate(tensors_list):
            if key in tensors:
                scaled_tensor = tensors[key] * scales[i]
                if merged_tensor is None:
                    merged_tensor = scaled_tensor.clone()
                else:
                    merged_tensor += scaled_tensor
        
        if merged_tensor is not None:
            merged[key] = merged_tensor
    
    return merged


def validate_lora_file(lora_path: Path) -> bool:
    """
    Validate that a LoRA file is valid and loadable.
    
    Args:
        lora_path: Path to the LoRA file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if not lora_path.exists():
            print(f"LoRA file does not exist: {lora_path}")
            return False
        
        if not lora_path.suffix == ".safetensors":
            print(f"LoRA file is not a .safetensors file: {lora_path}")
            return False
        
        if not ML_LIBRARIES_AVAILABLE:
            # Basic validation without loading
            print(f"LoRA file exists and has correct extension: {lora_path}")
            return True
        
        # Try to load the file
        tensors = load_file(lora_path)
        
        # Check if it has LoRA structure
        has_lora_keys = any(key.endswith(("up.weight", "down.weight")) for key in tensors.keys())
        if not has_lora_keys:
            print(f"File does not appear to contain LoRA weights: {lora_path}")
            return False
        
        print(f"LoRA file validation successful: {lora_path}")
        print(f"Contains {len(tensors)} tensors")
        return True
        
    except Exception as e:
        print(f"LoRA file validation failed for {lora_path}: {e}")
        return False


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a model directory.
    
    Args:
        model_path: Path to model directory
    
    Returns:
        Dictionary with model information
    """
    info = {
        "exists": False,
        "size_gb": 0,
        "components": [],
        "model_index": None
    }
    
    if not os.path.exists(model_path):
        return info
    
    info["exists"] = True
    
    # Calculate total size
    total_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except (OSError, IOError):
                pass  # Skip files we can't read
    
    info["size_gb"] = total_size / (1024**3)
    
    # Get components (subdirectories)
    try:
        info["components"] = [d for d in os.listdir(model_path) 
                             if os.path.isdir(os.path.join(model_path, d))]
    except (OSError, IOError):
        pass
    
    # Load model index if available
    model_index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(model_index_path):
        try:
            with open(model_index_path, 'r') as f:
                info["model_index"] = json.load(f)
        except (IOError, json.JSONDecodeError):
            pass
    
    return info


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
    
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_old_files(directory: Path, max_age_days: int = 7, pattern: str = "*.safetensors"):
    """
    Clean up old files in a directory based on modification time.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days before files are deleted
        pattern: File pattern to match (glob pattern)
    """
    if not directory.exists():
        return
    
    import time
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    deleted_count = 0
    for file_path in directory.glob(pattern):
        try:
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                deleted_count += 1
                print(f"Deleted old file: {file_path.name}")
        except (OSError, IOError) as e:
            print(f"Failed to delete {file_path}: {e}")
    
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old files from {directory}")


def check_gpu_memory() -> Dict[str, Any]:
    """
    Check GPU memory usage and availability for CUDA, MPS, and other backends.
    
    Returns:
        Dictionary with GPU memory information
    """
    info = {
        "available": False,
        "backend": "none",
        "device_count": 0,
        "total_memory": 0,
        "allocated_memory": 0,
        "cached_memory": 0,
        "free_memory": 0,
        "device_name": "Unknown"
    }
    
    try:
        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            info["available"] = True
            info["backend"] = "cuda"
            info["device_count"] = torch.cuda.device_count()
            
            if info["device_count"] > 0:
                info["device_name"] = torch.cuda.get_device_name(0)
                info["total_memory"] = torch.cuda.get_device_properties(0).total_memory
                info["allocated_memory"] = torch.cuda.memory_allocated(0)
                info["cached_memory"] = torch.cuda.memory_reserved(0)
                info["free_memory"] = info["total_memory"] - info["allocated_memory"]
        
        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["available"] = True
            info["backend"] = "mps"
            info["device_count"] = 1
            info["device_name"] = "Apple Silicon GPU"
            
            # MPS doesn't have direct memory query, estimate based on system
            import psutil
            # Apple Silicon uses unified memory
            total_memory = psutil.virtual_memory().total
            info["total_memory"] = total_memory // 2  # Rough estimate for GPU portion
            
            # MPS memory tracking (if available)
            if hasattr(torch.mps, 'current_allocated_memory'):
                info["allocated_memory"] = torch.mps.current_allocated_memory()
            
            info["free_memory"] = info["total_memory"] - info["allocated_memory"]
        
        # Convert to GB for readability
        for key in ["total_memory", "allocated_memory", "cached_memory", "free_memory"]:
            if key in info and info[key] > 0:
                info[f"{key}_gb"] = info[key] / (1024**3)
    
    except Exception as e:
        print(f"Failed to get GPU memory info: {e}")
    
    return info


def log_system_info():
    """Log system and GPU information for debugging."""
    print("=" * 50)
    print("System Information")
    print("=" * 50)
    
    # Platform info
    import platform
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Python info
    print(f"Python: {platform.python_version()}")
    
    # GPU info
    gpu_info = check_gpu_memory()
    if gpu_info["available"]:
        backend = gpu_info["backend"].upper()
        device_name = gpu_info.get("device_name", "Unknown")
        print(f"GPU: {backend} - {device_name} ({gpu_info['device_count']} devices)")
        
        if "total_memory_gb" in gpu_info:
            total_gb = gpu_info['total_memory_gb']
            free_gb = gpu_info.get('free_memory_gb', 0)
            print(f"Memory: {total_gb:.1f}GB total, {free_gb:.1f}GB free")
    else:
        print("GPU: Not available - using CPU")
    
    # PyTorch info
    print(f"PyTorch: {torch.__version__}")
    
    # Backend availability
    backends = []
    if torch.cuda.is_available():
        backends.append(f"CUDA {torch.version.cuda}")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        backends.append("MPS (Apple Silicon)")
    if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.enabled:
        backends.append("MKLDNN")
    
    if backends:
        print(f"Available backends: {', '.join(backends)}")
    else:
        print("Available backends: CPU only")
    
    # Model cache info
    cache_dirs = ["./sdxl-cache", "./refiner-cache", "./safety-cache"]
    total_cache_size = 0
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            info = get_model_info(cache_dir)
            size_gb = info['size_gb']
            total_cache_size += size_gb
            print(f"{cache_dir}: {size_gb:.1f}GB")
        else:
            print(f"{cache_dir}: Not downloaded")
    
    if total_cache_size > 0:
        print(f"Total cache size: {total_cache_size:.1f}GB")
    
    print("=" * 50)


# Aspect ratio mapping from flux format to SDXL dimensions
ASPECT_RATIO_MAP = {
    "1:1": (1024, 1024),    # Square - SDXL default
    "2:3": (832, 1248),     # Portrait 
    "3:2": (1248, 832),     # Landscape
    "3:4": (896, 1152),     # Portrait
    "4:3": (1152, 896),     # Landscape
    "4:5": (896, 1120),     # Portrait
    "5:4": (1120, 896),     # Landscape
    "9:16": (768, 1344),    # Vertical (phone)
    "16:9": (1344, 768),    # Horizontal (widescreen)
}

# Megapixels to dimension mapping (approximate)
MEGAPIXEL_MAP = {
    "0.25": (512, 512),     # 0.26 MP
    "0.5": (704, 704),      # 0.50 MP  
    "1": (1024, 1024),      # 1.05 MP - SDXL default
    "2": (1408, 1408),      # 1.98 MP
}

# Output format mapping
OUTPUT_FORMAT_EXTENSIONS = {
    "webp": "webp",
    "png": "png", 
    "jpg": "jpeg",
    "jpeg": "jpeg"
}


def parse_aspect_ratio(aspect_ratio: str, megapixels: str = "1") -> tuple[int, int]:
    """
    Parse aspect ratio string and return (width, height) for SDXL.
    
    Args:
        aspect_ratio: Aspect ratio in format "width:height" (e.g., "16:9")
        megapixels: Target megapixels as string
    
    Returns:
        Tuple of (width, height) in pixels
    """
    # Use predefined mapping if available
    if aspect_ratio in ASPECT_RATIO_MAP:
        base_width, base_height = ASPECT_RATIO_MAP[aspect_ratio]
        
        # Adjust for megapixels if not "1"
        if megapixels != "1" and megapixels in MEGAPIXEL_MAP:
            mp_width, mp_height = MEGAPIXEL_MAP[megapixels]
            # Scale proportionally while maintaining aspect ratio
            scale_factor = (mp_width * mp_height) / (base_width * base_height)
            scale_factor = scale_factor ** 0.5
            
            width = int(base_width * scale_factor)
            height = int(base_height * scale_factor)
            
            # Round to multiples of 8 (SDXL requirement)
            width = ((width + 7) // 8) * 8
            height = ((height + 7) // 8) * 8
            
            return width, height
        
        return base_width, base_height
    
    # Parse custom aspect ratio
    try:
        if ":" in aspect_ratio:
            w_ratio, h_ratio = map(float, aspect_ratio.split(":"))
        else:
            # Assume it's a decimal (e.g., "1.5" for 3:2)
            ratio = float(aspect_ratio)
            w_ratio, h_ratio = ratio, 1.0
        
        # Calculate dimensions based on megapixels
        if megapixels in MEGAPIXEL_MAP:
            target_width, target_height = MEGAPIXEL_MAP[megapixels]
            target_pixels = target_width * target_height
        else:
            target_pixels = 1024 * 1024  # Default 1MP
        
        # Calculate width and height maintaining aspect ratio
        aspect = w_ratio / h_ratio
        height = int((target_pixels / aspect) ** 0.5)
        width = int(height * aspect)
        
        # Round to multiples of 8
        width = ((width + 7) // 8) * 8
        height = ((height + 7) // 8) * 8
        
        return width, height
        
    except (ValueError, ZeroDivisionError):
        # Default to 1:1 if parsing fails
        print(f"Warning: Could not parse aspect ratio '{aspect_ratio}', using 1:1")
        return ASPECT_RATIO_MAP["1:1"]


def validate_dimensions(width: int, height: int, max_resolution: int = 2048) -> tuple[int, int]:
    """
    Validate and adjust dimensions for SDXL compatibility.
    
    Args:
        width: Target width
        height: Target height
        max_resolution: Maximum allowed resolution on either dimension
    
    Returns:
        Validated (width, height) tuple
    """
    # Ensure multiples of 8
    width = ((width + 7) // 8) * 8
    height = ((height + 7) // 8) * 8
    
    # Clamp to reasonable bounds for SDXL
    min_size = 512
    width = max(min_size, min(width, max_resolution))
    height = max(min_size, min(height, max_resolution))
    
    return width, height


def get_output_format_extension(output_format: str) -> str:
    """Get file extension for output format."""
    return OUTPUT_FORMAT_EXTENSIONS.get(output_format.lower(), "webp")


# Constants for model URLs and paths - keeping them local to this project
SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"

# Model download URLs
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-fix-1.0.tar"
REFINER_URL = "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

# Local models directory for testing
LOCAL_MODELS_DIR = "./local-models"

# Local custom models configuration for development/testing
LOCAL_CUSTOM_MODELS = {
    "illustrious-xl": {
        "type": "unet_only",
        "url": "https://huggingface.co/drawhisper/illustrious-xl/resolve/main/illustrious-xl.safetensors",
        "description": "Illustrious XL - High-quality anime/illustration generation (Default)",
        "recommended_cfg": 6.0,
        "recommended_steps": 28,
        "recommended_negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    "sdxl-base": {
        "type": "complete",
        "url": "https://weights.replicate.delivery/default/sdxl/sdxl-vae-fix-1.0.tar",
        "description": "Standard SDXL 1.0 (Fallback option)",
    }
}

# Production custom models (will be updated when deployed to Replicate)
CUSTOM_MODELS = {
    "default": {
        "type": "complete",
        "url": "https://weights.replicate.delivery/default/sdxl/sdxl-vae-fix-1.0.tar",
        "description": "Default SDXL 1.0",
    },
    # Add Replicate-hosted models here after deployment
}

def get_model_config_local(model_name: str = "default"):
    """Get model configuration for local development."""
    # Check if running locally vs production
    is_local = os.path.exists(LOCAL_MODELS_DIR)
    
    if is_local and model_name in LOCAL_CUSTOM_MODELS:
        return LOCAL_CUSTOM_MODELS[model_name]
    elif model_name in CUSTOM_MODELS:
        return CUSTOM_MODELS[model_name]
    else:
        print(f"Warning: Unknown model '{model_name}', using default")
        return LOCAL_CUSTOM_MODELS.get("illustrious-xl", CUSTOM_MODELS.get("illustrious-xl", LOCAL_CUSTOM_MODELS["illustrious-xl"]))

def download_custom_model_local(model_name: str, cache_dir: str = "./custom-models"):
    """Download or locate custom model for local development and production."""
    config = get_model_config_local(model_name)
    ensure_directory(cache_dir)
    
    # Check if local path exists first (for local development)
    if "local_path" in config and os.path.exists(config["local_path"]):
        print(f"Using local model: {config['local_path']}")
        return config["local_path"]
    
    # Check for pre-downloaded model in production (cog.yaml downloads to /src/custom-models)
    production_path = f"/src/custom-models/{model_name}-unet.safetensors"
    if os.path.exists(production_path):
        print(f"Using pre-cached model: {production_path}")
        return production_path
    
    # Download from URL (fallback for local development or if pre-cache failed)
    if config["type"] == "unet_only":
        model_path = os.path.join(cache_dir, f"{model_name}-unet.safetensors")
        if not os.path.exists(model_path):
            if "url" in config:
                print(f"Downloading {model_name} UNet from {config['url']}")
                
                # Use pget with retry logic (Replicate's recommended approach)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"Attempt {attempt + 1}/{max_retries}: Downloading with pget...")
                        subprocess.check_call([
                            "pget", 
                            "--timeout", "300",  # 5 minute timeout
                            config["url"], 
                            model_path
                        ])
                        print(f"Successfully downloaded {model_name} to {model_path}")
                        break
                    except subprocess.CalledProcessError as e:
                        if attempt < max_retries - 1:
                            print(f"Download attempt {attempt + 1} failed, retrying...")
                            import time
                            time.sleep(10)  # Wait before retry
                        else:
                            raise RuntimeError(f"Failed to download {model_name} after {max_retries} attempts: {e}")
            else:
                raise ValueError(f"No local path or URL found for model: {model_name}")
        return model_path
    else:
        # Complete model package
        model_path = os.path.join(cache_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} complete model...")
            download_weights(config["url"], model_path)
        return model_path