import base64
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

from cog import Secret
from huggingface_hub import HfApi, hf_hub_download, login, logout

# Detect environment and set appropriate cache directory
import os
if os.path.exists("/src"):
    # Running in Docker/cog environment
    DEFAULT_CACHE_BASE_DIR = Path("/src/sdxl-lora-cache")
else:
    # Running locally
    DEFAULT_CACHE_BASE_DIR = Path("./sdxl-lora-cache")


class SDXLLoRACache:
    """
    Enhanced LRU cache for SDXL LoRA weights, adapted from flux-dev-lora implementation.
    Supports HuggingFace, CivitAI, and direct URL downloads with intelligent caching.
    """
    def __init__(
        self, min_disk_free: int = 5 * (2**30), base_dir: Path = DEFAULT_CACHE_BASE_DIR
    ):
        self.min_disk_free = min_disk_free
        self.base_dir = base_dir
        self.hits = 0
        self.misses = 0

        # Least Recently Used (LRU) cache for paths
        self.lru_paths = deque()
        base_dir.mkdir(parents=True, exist_ok=True)

    def ensure(
        self,
        url: str,
        hf_api_token: Optional[Secret] = None,
        civitai_api_token: Optional[Secret] = None,
    ) -> Path:
        """
        Ensure LoRA weights are available locally, downloading if necessary.
        Returns path to the cached LoRA file.
        """
        path = self._lora_path(url)

        if path in self.lru_paths:
            # Move to end of LRU (mark as recently used)
            self.hits += 1
            self.lru_paths.remove(path)
        elif not path.exists():
            self.misses += 1

            # Free up space if needed
            while not self._has_enough_space() and len(self.lru_paths) > 0:
                self._remove_least_recent()

            download_lora_weights(
                url,
                path,
                hf_api_token=hf_api_token,
                civitai_api_token=civitai_api_token,
            )

        self.lru_paths.append(path)
        return path

    def cache_info(self) -> str:
        return f"SDXLLoRACache(hits={self.hits}, misses={self.misses}, base_dir='{self.base_dir}', currsize={len(self.lru_paths)})"

    def _remove_least_recent(self) -> None:
        oldest = self.lru_paths.popleft()
        print(f"Removing oldest LoRA from cache: {oldest}")
        if oldest.exists():
            oldest.unlink()

    def _has_enough_space(self) -> bool:
        disk_usage = shutil.disk_usage(self.base_dir)
        free = disk_usage.free
        return free >= self.min_disk_free

    def _lora_path(self, url: str) -> Path:
        hashed_url = hashlib.sha256(url.encode()).hexdigest()
        short_hash = hashed_url[:16]
        return self.base_dir / f"{short_hash}.safetensors"


@contextmanager
def logged_in_to_huggingface(
    token: Optional[Secret] = None, add_to_git_credential: bool = False
):
    """Context manager for temporary Hugging Face login."""
    try:
        if token is not None:
            print("Logging into HuggingFace with provided token...")
            login(
                token=token.get_secret_value(),
                add_to_git_credential=add_to_git_credential,
            )
            print("HuggingFace login successful!")
        yield
    finally:
        logout()
        print("Logged out of HuggingFace.")


def download_lora_weights(
    url: str,
    path: Path,
    hf_api_token: Optional[Secret] = None,
    civitai_api_token: Optional[Secret] = None,
):
    """Download LoRA weights from various sources."""
    download_url = make_lora_download_url(url, civitai_api_token=civitai_api_token)
    download_lora_weights_url(download_url, path, hf_api_token=hf_api_token)


def download_lora_weights_url(url: str, path: Path, hf_api_token: Optional[Secret] = None):
    """
    Download LoRA weights from a processed URL.
    Supports HuggingFace, CivitAI, direct URLs, and various archive formats.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading LoRA weights...")
    start_time = time.time()

    # Handle HuggingFace URLs - adapted from flux weights.py:127-159
    if m := re.match(
        r"^(?:https?://)?huggingface\.co/([^/]+)/([^/]+)(?:/([^/]+\.safetensors))?/?$",
        url,
    ):
        if len(m.groups()) == 2:
            owner, model_name = m.groups()
            lora_weights = None
        else:
            owner, model_name, lora_weights = m.groups()

        try:
            with logged_in_to_huggingface(hf_api_token):
                if lora_weights is None:
                    repo_id = f"{owner}/{model_name}"
                    files = HfApi().list_repo_files(repo_id)
                    sft_files = [file for file in files if ".safetensors" in file]
                    if len(sft_files) == 1:
                        lora_weights = sft_files[0]
                    elif len(sft_files) > 1:
                        # Prefer common SDXL LoRA filenames
                        preferred_names = [
                            "adapter_model.safetensors", 
                            "pytorch_lora_weights.safetensors",
                            "lora_weights.safetensors"
                        ]
                        for preferred in preferred_names:
                            if preferred in sft_files:
                                lora_weights = preferred
                                break
                        else:
                            lora_weights = sft_files[0]  # fallback to first file
                    else:
                        raise ValueError(
                            f"No *.safetensors file found in HuggingFace repo {repo_id}"
                        )

                safetensors_path = hf_hub_download(
                    repo_id=f"{owner}/{model_name}",
                    filename=lora_weights,
                )
                shutil.copy(Path(safetensors_path), path)
                print(f"Downloaded {lora_weights} from HuggingFace to {path}")
        except Exception as e:
            raise ValueError(f"Failed to download from HuggingFace: {e}")

    # Handle data URLs
    elif url.startswith("data:"):
        download_data_url(url, path)
    
    # Handle tar files
    elif url.endswith(".tar"):
        download_safetensors_tarball(url, path)
    
    # Handle safetensors files (including CivitAI) - adapted from flux weights.py:164-169
    elif (
        url.endswith(".safetensors")
        or "://civitai.com/api/download" in url
        or ".safetensors?" in url
    ):
        download_safetensors(url, path)
    
    # Handle Replicate weights
    elif url.endswith("/_weights"):
        download_safetensors_tarball(url, path)
    else:
        raise ValueError("URL must end with either .tar or .safetensors, or be a valid HuggingFace/CivitAI URL")

    print(f"Downloaded LoRA weights in {time.time() - start_time:.2f}s")


def find_safetensors(directory: Path) -> list[Path]:
    """Find all .safetensors files in a directory recursively."""
    safetensors_paths = []
    for root, _, files in os.walk(directory):
        root = Path(root)
        for filename in files:
            path = root / filename
            if path.suffix == ".safetensors":
                safetensors_paths.append(path)
    return safetensors_paths


def download_safetensors_tarball(url: str, path: Path):
    """Download and extract safetensors from a tarball - adapted from flux weights.py:189-208."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        extract_dir = temp_dir / "weights"

        try:
            subprocess.run(
                ["pget", "-v", "-x", url, extract_dir], check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download tarball: {e}")

        safetensors_paths = find_safetensors(extract_dir)
        if not safetensors_paths:
            raise ValueError("No .safetensors file found in tarball")
        if len(safetensors_paths) > 1:
            # Prefer LoRA-related files
            lora_files = [p for p in safetensors_paths if "lora" in p.name.lower()]
            if lora_files:
                safetensors_path = lora_files[0]
            else:
                safetensors_path = safetensors_paths[0]
        else:
            safetensors_path = safetensors_paths[0]

        shutil.move(safetensors_path, path)


def download_data_url(url: str, path: Path):
    """Download safetensors from a base64 data URL - adapted from flux weights.py:211-226."""
    _, encoded = url.split(",", 1)
    data = base64.b64decode(encoded)

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(fileobj=BytesIO(data), mode="r:*") as tar:
            tar.extractall(path=temp_dir)

        safetensors_paths = find_safetensors(Path(temp_dir))
        if not safetensors_paths:
            raise ValueError("No .safetensors file found in data URI")
        if len(safetensors_paths) > 1:
            raise ValueError("Multiple .safetensors files found in data URI")
        safetensors_path = safetensors_paths[0]

        shutil.move(safetensors_path, path)


def download_safetensors(url: str, path: Path):
    """Download a single safetensors file directly - adapted from flux weights.py:229-264."""
    try:
        # Don't leak CivitAI API keys in logs
        if "token=" in url:
            print(f"Downloading LoRA weights from {url.split('token=')[0]}token=***")
        else:
            print(f"Downloading LoRA weights from {url}")

        result = subprocess.run(
            ["pget", url, str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            error_output = result.stderr or ""
            if "401" in error_output:
                raise RuntimeError(
                    "Authorization failed. Please check if an API key is needed."
                )
            if "404" in error_output:
                if "civitai" in url:
                    raise RuntimeError(
                        "LoRA not found on CivitAI. Check the model ID in the URL."
                    )
                raise RuntimeError(
                    "LoRA not found at the supplied URL. Please check the URL."
                )
            raise RuntimeError(f"Failed to download LoRA file: {error_output}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download LoRA file: {e}")


def make_lora_download_url(url: str, civitai_api_token: Optional[Secret] = None) -> str:
    """
    Convert various URL formats to downloadable URLs.
    Adapted from flux weights.py:267-311.
    """
    if url.startswith("data:"):
        return url
    
    # HuggingFace URLs
    if m := re.match(
        r"^(?:https?://)?huggingface\.co/([^/]+)/([^/]+)(?:/([^/]+\.safetensors))?/?$",
        url,
    ):
        if len(m.groups()) not in [2, 3]:
            raise ValueError(
                "Invalid HuggingFace URL. Expected format: huggingface.co/<owner>/<model-name>[/<lora-weights-file.safetensors>]"
            )
        return url
    
    # CivitAI model URLs
    if m := re.match(r"^(?:https?://)?civitai\.com/models/(\d+)(?:/[^/?]+)?/?$", url):
        model_id = m.groups()[0]
        return make_civitai_download_url(model_id, civitai_api_token)
    
    # CivitAI API URLs
    if m := re.match(r"^((?:https?://)?civitai\.com/api/download/models/.*)$", url):
        return url
    
    # Direct safetensors URLs
    if m := re.match(r"^(https?://.*\.safetensors(\?.*)?)$", url):
        return url
    
    # Replicate model URLs (but avoid false matches with other platforms)
    if m := re.match(r"^(?:https?://replicate.com/)?([^/]+)/([^/]+)/?$", url):
        owner, model_name = m.groups()
        # Don't match URLs that look like other platforms
        if not ("huggingface.co" in url or "civitai.com" in url):
            return f"https://replicate.com/{owner}/{model_name}/_weights"
    
    # Replicate version URLs
    if m := re.match(
        r"^(?:https?://replicate.com/)?([^/]+)/([^/]+)/(?:versions/)?([^/]+)/?$", url
    ):
        owner, model_name, version_id = m.groups()
        # Don't match URLs that look like other platforms
        if not ("huggingface.co" in url or "civitai.com" in url):
            return f"https://replicate.com/{owner}/{model_name}/versions/{version_id}/_weights"
    
    # Replicate delivery URLs
    if m := re.match(r"^(https?://replicate.delivery/.*\.tar)$", url):
        return m.groups()[0]

    # Error handling for common mistakes
    if "huggingface.co" in url:
        raise ValueError(
            "Failed to parse HuggingFace URL. Expected huggingface.co/<owner>/<model-name>[/<lora-weights-file.safetensors>]"
        )
    if "civitai.com" in url:
        raise ValueError(
            "Failed to parse CivitAI URL. Expected civitai.com/models/<id>[/<model-name>]"
        )
    
    raise ValueError(
        """Failed to parse URL. Expected either:
* HuggingFace URL in the format huggingface.co/<owner>/<model-name>[/<lora-weights-file.safetensors>]
* CivitAI URL in the format civitai.com/models/<id>[/<model-name>]
* Replicate model in the format <owner>/<username> or <owner>/<username>/<version>
* Direct .safetensors URLs from the Internet"""
    )


def make_civitai_download_url(
    model_id: str, civitai_api_token: Optional[Secret] = None
) -> str:
    """Create CivitAI download URL with optional API token - adapted from flux weights.py:324-329."""
    base_url = f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
    if civitai_api_token is None:
        return base_url
    return f"{base_url}&token={civitai_api_token.get_secret_value()}"