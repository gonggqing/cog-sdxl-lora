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
    from PIL import Image
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False

def download_from_huggingface_cli(repo_id: str, cache_dir: str = "./hf-models") -> str:
    """
    Download a complete HuggingFace model using HF CLI equivalent in Python.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "drawhisper/illustrious-xl")
        cache_dir: Local cache directory
    
    Returns:
        Path to the downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
        
        print(f"Downloading complete model {repo_id} using HF CLI equivalent...")
        ensure_directory(cache_dir)
        
        # Use snapshot_download to get the entire model (equivalent to `hf download`)
        model_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, repo_id.replace('/', '--')),
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
        )
        
        print(f"HuggingFace model downloaded to: {model_path}")
        return model_path
        
    except ImportError:
        raise RuntimeError("huggingface_hub is required for HuggingFace downloads. Install with: pip install huggingface_hub")
    except Exception as e:
        raise RuntimeError(f"Failed to download {repo_id} using HF CLI: {e}")


def download_from_huggingface(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> str:
    """
    Download a file from HuggingFace using native caching.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "drawhisper/illustrious-xl")
        filename: File name to download (e.g., "illustrious-xl.safetensors")
        cache_dir: Optional cache directory (defaults to HF default cache)
    
    Returns:
        Path to the downloaded file in HF cache
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"Downloading {filename} from HuggingFace repo: {repo_id}")
        
        kwargs = {
            "repo_id": repo_id,
            "filename": filename,
        }
        
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
            
        local_file_path = hf_hub_download(**kwargs)
        print(f"HuggingFace file downloaded to: {local_file_path}")
        return local_file_path
        
    except ImportError:
        raise RuntimeError("huggingface_hub is required for HuggingFace downloads. Install with: pip install huggingface_hub")
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename} from HuggingFace repo {repo_id}: {e}")


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


def check_lora_compatibility(pipeline, lora_path: Path) -> bool:
    """
    Check if a LoRA is compatible with the current pipeline architecture.
    
    Args:
        pipeline: The diffusers pipeline
        lora_path: Path to the LoRA .safetensors file
    
    Returns:
        True if compatible, False otherwise
    """
    try:
        from safetensors.torch import load_file
        
        # Load LoRA state dict
        lora_state_dict = load_file(lora_path)
        
        # Get current model state dict
        model_state_dict = pipeline.unet.state_dict()
        
        # Check for critical shape mismatches
        incompatible_layers = []
        for key, lora_tensor in lora_state_dict.items():
            # Extract the corresponding model key (remove LoRA-specific suffixes)
            model_key = key.replace('.lora_up.weight', '.weight').replace('.lora_down.weight', '.weight')
            model_key = model_key.replace('.lora_magnitude_vector', '').split('.default_')[0]
            
            if model_key in model_state_dict:
                model_shape = model_state_dict[model_key].shape
                lora_shape = lora_tensor.shape
                
                # Check for critical dimension mismatches
                if 'lora_magnitude_vector' in key and len(lora_shape) > 0:
                    expected_dim = model_shape[0] if len(model_shape) > 0 else 0
                    if lora_shape[-1] != expected_dim and lora_shape[0] != 1:
                        incompatible_layers.append(f"{key}: LoRA {lora_shape} vs Model {model_shape}")
        
        if incompatible_layers:
            print(f"⚠️ LoRA compatibility issues detected:")
            for issue in incompatible_layers[:3]:  # Show first 3 issues
                print(f"   {issue}")
            if len(incompatible_layers) > 3:
                print(f"   ... and {len(incompatible_layers) - 3} more")
            return False
            
        return True
        
    except Exception as e:
        print(f"⚠️ Could not check LoRA compatibility: {e}")
        return True  # Assume compatible if we can't check


def load_lora_weights_to_pipeline(pipeline, lora_path: Path, adapter_name: str = "default", lora_scale: float = 1.0, strict: bool = False):
    """
    Load LoRA weights into a diffusers pipeline with enhanced error handling.
    
    Args:
        pipeline: The diffusers pipeline to load LoRA into
        lora_path: Path to the LoRA .safetensors file
        adapter_name: Name for the LoRA adapter
        lora_scale: Scale factor for the LoRA weights
        strict: If True, fail on any errors. If False, try multiple fallback methods
    """
    if not ML_LIBRARIES_AVAILABLE:
        raise RuntimeError("ML libraries (diffusers, safetensors) not available. Install requirements.txt dependencies.")
    
    print(f"Loading LoRA weights from {lora_path} with scale {lora_scale}")
    
    # Check compatibility first
    if not check_lora_compatibility(pipeline, lora_path):
        if strict:
            raise RuntimeError(f"LoRA {lora_path} is incompatible with current model architecture")
        else:
            print("⚠️ LoRA may be incompatible, but attempting to load anyway...")
    
    # Method 1: Standard diffusers loading
    try:
        pipeline.load_lora_weights(
            str(lora_path),
            adapter_name=adapter_name,
        )
        
        # Set the LoRA scale
        pipeline.set_adapters([adapter_name], adapter_weights=[lora_scale])
        
        print(f"✅ Successfully loaded LoRA '{adapter_name}' with scale {lora_scale}")
        return
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Method 1 failed: {error_msg}")
        
        # If it's a shape mismatch and we're not in strict mode, try workarounds
        if "size mismatch" in error_msg and not strict:
            print("🔧 Attempting shape mismatch workarounds...")
            
            # Method 2: Try loading with ignore_mismatched_sizes (if supported)
            try:
                print("Trying with relaxed loading...")
                pipeline.load_lora_weights(
                    str(lora_path),
                    adapter_name=adapter_name,
                    ignore_mismatched_sizes=True,
                )
                pipeline.set_adapters([adapter_name], adapter_weights=[lora_scale])
                print(f"✅ Successfully loaded LoRA '{adapter_name}' with relaxed loading")
                return
            except Exception as e2:
                print(f"❌ Method 2 failed: {e2}")
            
            # Method 3: Try alternative loading approach
            try:
                print("Trying alternative LoRA loading method...")
                pipeline.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
                if hasattr(pipeline, 'fuse_lora'):
                    pipeline.fuse_lora(lora_scale=lora_scale)
                elif hasattr(pipeline, 'set_adapters'):
                    pipeline.set_adapters([adapter_name], adapter_weights=[lora_scale])
                print(f"✅ Successfully loaded LoRA using alternative method")
                return
            except Exception as e3:
                print(f"❌ Method 3 failed: {e3}")
            
            # Method 4: Manual loading with filtering (last resort)
            try:
                print("Trying manual loading with shape filtering...")
                load_lora_with_shape_filtering(pipeline, lora_path, adapter_name, lora_scale)
                print(f"✅ Successfully loaded LoRA using filtered loading")
                return
            except Exception as e4:
                print(f"❌ Method 4 failed: {e4}")
        
        # If all methods failed
        if strict:
            raise RuntimeError(f"Failed to load LoRA after trying all methods. Final error: {e}")
        else:
            print(f"⚠️ Could not load LoRA {lora_path}. Skipping...")
            return


def load_lora_with_shape_filtering(pipeline, lora_path: Path, adapter_name: str, lora_scale: float):
    """
    Load LoRA with manual shape filtering to skip incompatible layers.
    This is a last resort method for incompatible LoRAs.
    """
    from safetensors.torch import load_file
    import torch
    
    print("🔧 Loading LoRA with shape filtering (experimental)...")
    
    # Load LoRA state dict
    lora_state_dict = load_file(lora_path)
    model_state_dict = pipeline.unet.state_dict()
    
    # Filter out incompatible weights
    filtered_state_dict = {}
    skipped_layers = []
    
    for key, tensor in lora_state_dict.items():
        try:
            # Try to find corresponding model layer
            model_key = key.replace('.lora_up.weight', '.weight').replace('.lora_down.weight', '.weight')
            model_key = model_key.replace('.lora_magnitude_vector', '').split('.default_')[0]
            
            if model_key in model_state_dict:
                model_tensor = model_state_dict[model_key]
                
                # Check compatibility
                if 'lora_magnitude_vector' in key:
                    # Special handling for magnitude vectors
                    expected_dim = model_tensor.shape[0]
                    if tensor.shape[-1] == expected_dim or tensor.shape[0] == 1:
                        filtered_state_dict[key] = tensor
                    else:
                        skipped_layers.append(key)
                else:
                    # For regular LoRA weights, be more permissive
                    filtered_state_dict[key] = tensor
            else:
                filtered_state_dict[key] = tensor
                
        except Exception:
            skipped_layers.append(key)
    
    if skipped_layers:
        print(f"⚠️ Skipped {len(skipped_layers)} incompatible layers")
    
    if not filtered_state_dict:
        raise RuntimeError("No compatible LoRA layers found")
    
    # Create a temporary file with filtered weights
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp_file:
        from safetensors.torch import save_file
        save_file(filtered_state_dict, tmp_file.name)
        
        # Try loading the filtered LoRA
        pipeline.load_lora_weights(tmp_file.name, adapter_name=adapter_name)
        pipeline.set_adapters([adapter_name], adapter_weights=[lora_scale])
        
        # Clean up
        import os
        os.unlink(tmp_file.name)


def unload_lora_weights_from_pipeline(pipeline, adapter_name: str = "default"):
    """
    Unload LoRA weights from a diffusers pipeline.
    
    Args:
        pipeline: The diffusers pipeline to unload LoRA from
        adapter_name: Name of the LoRA adapter to unload
    """
    try:
        if hasattr(pipeline, 'delete_adapters'):
            pipeline.delete_adapters([adapter_name])
            print(f"✅ Successfully unloaded LoRA adapter '{adapter_name}'")
        elif hasattr(pipeline, 'unfuse_lora'):
            pipeline.unfuse_lora()
            print("✅ Successfully unfused LoRA weights")
        else:
            print("⚠️ Pipeline does not support LoRA unloading")
    except Exception as e:
        print(f"⚠️ Failed to unload LoRA: {e}")


# Legacy function for backward compatibility
def load_lora_weights_to_unet(unet, lora_path: Path, device: str = "cuda") -> Dict[str, Any]:
    """
    Legacy function - use load_lora_weights_to_pipeline instead.
    This function is kept for backward compatibility but is deprecated.
    """
    print("⚠️ Warning: load_lora_weights_to_unet is deprecated. Use load_lora_weights_to_pipeline instead.")
    raise NotImplementedError(
        "Manual UNet LoRA loading is deprecated. "
        "Use load_lora_weights_to_pipeline with the full pipeline instead."
    )


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
            try:
                import psutil
                # Apple Silicon uses unified memory
                total_memory = psutil.virtual_memory().total
                info["total_memory"] = total_memory // 2  # Rough estimate for GPU portion
            except ImportError:
                # psutil not available, use default estimate
                info["total_memory"] = 8 * 1024**3  # 8GB default estimate
            
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

# Model download URLs
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-fix-1.0.tar"
REFINER_URL = "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

# Local models directory for testing
LOCAL_MODELS_DIR = "./local-models"

# Local custom models configuration for development/testing
LOCAL_CUSTOM_MODELS = {
    "animagine-xl-4.0": {
        "type": "complete",
        "source": "huggingface_cli",
        "repo_id": "cagliostrolab/animagine-xl-4.0",
        "description": " High-quality base model for anime/illustration generation (Default)",
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
    if "filename" in config:
        production_path = f"/src/custom-models/{config['filename']}"
    else:
        production_path = f"/src/custom-models/{model_name}.safetensors"
    
    if os.path.exists(production_path):
        print(f"Using pre-cached model: {production_path}")
        return production_path
    
    # Try HuggingFace CLI download first if it's a HF CLI model
    if config.get("source") == "huggingface_cli" and "repo_id" in config:
        try:
            print(f"Attempting HuggingFace CLI download for {model_name}...")
            hf_cached_path = download_from_huggingface_cli(
                repo_id=config["repo_id"],
                cache_dir=cache_dir
            )
            print(f"Using HuggingFace CLI cached model: {hf_cached_path}")
            return hf_cached_path
        except Exception as e:
            print(f"HuggingFace CLI download failed: {e}")
            print("Model not available via HF CLI, skipping...")
    
    # Try HuggingFace single file download if it's a HF model
    elif config.get("source") == "huggingface" and "repo_id" in config and "filename" in config:
        try:
            print(f"Attempting HuggingFace single file download for {model_name}...")
            hf_cached_path = download_from_huggingface(
                repo_id=config["repo_id"],
                filename=config["filename"]
            )
            print(f"Using HuggingFace cached model: {hf_cached_path}")
            return hf_cached_path
        except Exception as e:
            print(f"HuggingFace download failed: {e}")
            print("Falling back to direct URL download...")
    
    # Download from URL (fallback for local development or if HF download failed)
    if config["type"] == "safetensors_only":
        # Use original filename if available, otherwise generate one
        if "filename" in config:
            filename = config["filename"]
        else:
            filename = f"{model_name}.safetensors"
        model_path = os.path.join(cache_dir, filename)
        if not os.path.exists(model_path):
            if "url" in config:
                print(f"Downloading {model_name} from {config['url']}")
                
                # Use pget with retry logic (Replicate's recommended approach)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"Attempt {attempt + 1}/{max_retries}: Downloading with pget...")
                        subprocess.check_call([
                            "pget", 
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