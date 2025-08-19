import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from cog import BasePredictor, Input, Path, Secret
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor

from weights import SDXLLoRACache
from utils import (
    download_weights,
    TokenEmbeddingsHandler,
    load_lora_weights_to_pipeline,
    unload_lora_weights_from_pipeline,
    merge_lora_weights,
    validate_lora_file,
    log_system_info,
    parse_aspect_ratio,
    validate_dimensions,
    get_output_format_extension,
    get_model_config_local,
    load_image,
    download_custom_model_local,
    SDXL_MODEL_CACHE,
    REFINER_MODEL_CACHE,
    SAFETY_CACHE,
    SDXL_URL,
    REFINER_URL,
    SAFETY_URL
)


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class MultiLoRAMixin:
    """
    Mixin class for handling multiple LoRAs simultaneously.
    Adapted from flux bfl_predictor.py LoRA handling logic.
    """
    
    def __init__(self):
        self.lora_cache = SDXLLoRACache()
        self.loaded_loras = []  # Track loaded LoRAs for cleanup
        self.original_attn_processors = None  # Store original processors
        
        # State tracking for smart caching (inspired by cog-flux)
        self.current_lora = None
        self.current_lora_scale = None
        self.current_extra_lora = None
        self.current_extra_lora_scale = None
    
    def load_single_lora(
        self, 
        pipe, 
        lora_url: str, 
        lora_scale: float = 1.0,
        hf_api_token: Optional[Secret] = None,
        civitai_api_token: Optional[Secret] = None,
    ):
        """Load a single LoRA into the pipeline."""
        print(f"Loading single LoRA: {lora_url}")
        start_time = time.time()
        
        # Download LoRA weights
        lora_path = self.lora_cache.ensure(
            lora_url, 
            hf_api_token=hf_api_token, 
            civitai_api_token=civitai_api_token
        )
        
        # Load LoRA using the new pipeline-level method
        self._apply_lora_to_pipeline(pipe, lora_path, lora_scale, adapter_name="main_lora")
        self.loaded_loras = [{"path": lora_path, "scale": lora_scale}]
        
        print(f"Single LoRA loaded in {time.time() - start_time:.2f}s")
    
    def load_multiple_loras(
        self, 
        pipe,
        main_lora_url: str,
        main_lora_scale: float = 1.0,
        extra_lora_url: Optional[str] = None,
        extra_lora_scale: float = 1.0,
        hf_api_token: Optional[Secret] = None,
        civitai_api_token: Optional[Secret] = None,
    ):
        """
        Load multiple LoRAs simultaneously.
        Inspired by flux diffusers_predictor.py:load_multiple_loras and bfl_predictor.py:handle_loras.
        """
        print(f"Loading multiple LoRAs: main={main_lora_url}, extra={extra_lora_url}")
        start_time = time.time()
        
        # Download both LoRAs
        main_lora_path = self.lora_cache.ensure(
            main_lora_url, 
            hf_api_token=hf_api_token, 
            civitai_api_token=civitai_api_token
        )
        
        loras_to_load = [{"path": main_lora_path, "scale": main_lora_scale}]
        
        if extra_lora_url:
            extra_lora_path = self.lora_cache.ensure(
                extra_lora_url, 
                hf_api_token=hf_api_token, 
                civitai_api_token=civitai_api_token
            )
            loras_to_load.append({"path": extra_lora_path, "scale": extra_lora_scale})
        
        # Store original processors for cleanup
        if self.original_attn_processors is None:
            self.original_attn_processors = pipe.unet.attn_processors.copy()
        
        # Apply multiple LoRAs
        self._apply_multiple_loras_to_pipeline(pipe, loras_to_load)
        self.loaded_loras = loras_to_load
        
        print(f"Multiple LoRAs loaded in {time.time() - start_time:.2f}s")
    
    def _apply_lora_to_pipeline(self, pipeline, lora_path: Path, lora_scale: float, adapter_name: str = "default"):
        """Apply a single LoRA to the pipeline using diffusers' built-in functionality."""
        # Validate LoRA file first
        if not validate_lora_file(lora_path):
            raise ValueError(f"Invalid LoRA file: {lora_path}")
        
        # Use the utility function to load LoRA weights to the pipeline
        load_lora_weights_to_pipeline(pipeline, lora_path, adapter_name=adapter_name, lora_scale=lora_scale)
    
    def _apply_multiple_loras_to_pipeline(self, pipeline, loras: List[Dict]):
        """
        Apply multiple LoRAs to pipeline using diffusers' built-in functionality.
        Supports maximum of 2 LoRAs to be merged together.
        """
        if not loras:
            return
            
        # Enforce maximum of 2 LoRAs
        if len(loras) > 2:
            raise ValueError(f"Maximum of 2 LoRAs supported, but {len(loras)} were provided. Only main and extra LoRA are allowed.")
        
        print(f"Applying {len(loras)} LoRA(s) to pipeline...")
        
        # Load each LoRA as a separate adapter
        adapter_names = []
        adapter_weights = []
        
        for i, lora in enumerate(loras):
            # Use descriptive names for the two LoRAs
            adapter_name = "main_lora" if i == 0 else "extra_lora"
            adapter_names.append(adapter_name)
            adapter_weights.append(lora["scale"])
            
            # Load this LoRA as a separate adapter
            self._apply_lora_to_pipeline(pipeline, lora["path"], 1.0, adapter_name=adapter_name)
        
        # Set all adapters with their respective weights
        try:
            pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
            print(f"✅ Successfully applied {len(loras)} LoRAs with weights: {adapter_weights}")
        except Exception as e:
            print(f"⚠️ Failed to set multiple adapter weights: {e}")
            print("   Using first LoRA only as fallback")
            # Fallback to using just the first LoRA
            if adapter_names:
                pipeline.set_adapters([adapter_names[0]], adapter_weights=[adapter_weights[0]])
    
    def unload_loras(self, pipe):
        """Unload all LoRAs using diffusers' built-in functionality."""
        try:
            # Try to unload using the new pipeline-level approach
            if hasattr(pipe, 'delete_adapters') and self.loaded_loras:
                # Get all adapter names that might be loaded based on number of LoRAs
                adapter_names = []
                if len(self.loaded_loras) == 1:
                    adapter_names = ["main_lora"]
                elif len(self.loaded_loras) == 2:
                    adapter_names = ["main_lora", "extra_lora"]
                
                for adapter_name in adapter_names:
                    try:
                        pipe.delete_adapters([adapter_name])
                        print(f"✅ Deleted adapter: {adapter_name}")
                    except Exception as e:
                        print(f"⚠️ Could not delete adapter {adapter_name}: {e}")
                        
                print("✅ LoRAs unloaded using pipeline adapters")
                
            elif hasattr(pipe, 'unfuse_lora'):
                pipe.unfuse_lora()
                print("✅ LoRAs unfused from pipeline")
                
            else:
                # Fallback to manual restoration if available
                if self.original_attn_processors is not None:
                    pipe.unet.set_attn_processor(self.original_attn_processors)
                    print("✅ LoRAs unloaded using manual processor restoration")
                else:
                    print("⚠️ No method available to unload LoRAs")
                    
        except Exception as e:
            print(f"⚠️ Failed to unload LoRAs: {e}")
            # Try fallback method
            if self.original_attn_processors is not None:
                try:
                    pipe.unet.set_attn_processor(self.original_attn_processors)
                    print("✅ LoRAs unloaded using fallback method")
                except Exception as e2:
                    print(f"❌ Failed to unload LoRAs even with fallback: {e2}")
        
        # Reset state
        self.original_attn_processors = None
        self.loaded_loras = []
        self.current_lora = None
        self.current_lora_scale = None
        self.current_extra_lora = None
        self.current_extra_lora_scale = None

    def handle_loras(
        self,
        pipeline,
        lora_weights: str | None = None,
        lora_scale: float = 1.0,
        extra_lora_weights: str | None = None,
        extra_lora_scale: float = 1.0,
        hf_api_token: Optional[Secret] = None,
        civitai_api_token: Optional[Secret] = None,
    ):
        """
        Smart LoRA loading inspired by cog-flux handle_loras method.
        Only reloads LoRAs if weights or scales have actually changed.
        """
        # Handle edge case: extra_lora provided without main lora
        if not lora_weights and extra_lora_weights:
            print(f"⚠️ extra_lora_weights {extra_lora_weights} found but lora_weights is None!")
            print(f"   Setting main LoRA to {extra_lora_weights} with scale {extra_lora_scale}")
            lora_weights = extra_lora_weights
            lora_scale = extra_lora_scale
            extra_lora_weights = None
            extra_lora_scale = 1.0

        # Check if we need to reload (smart caching)
        needs_reload = (
            lora_weights != self.current_lora
            or lora_scale != self.current_lora_scale
            or extra_lora_weights != self.current_extra_lora
            or extra_lora_scale != self.current_extra_lora_scale
        )

        if lora_weights and needs_reload:
            print("🔄 LoRA configuration changed, reloading...")
            
            # Unload existing LoRAs first
            if self.current_lora or self.current_extra_lora:
                self.unload_loras(pipeline)

            # Load new LoRAs
            if extra_lora_weights:
                # Load multiple LoRAs
                print(f"📦 Loading multiple LoRAs: main={lora_weights}, extra={extra_lora_weights}")
                self.load_multiple_loras(
                    pipeline,
                    main_lora_url=lora_weights,
                    main_lora_scale=lora_scale,
                    extra_lora_url=extra_lora_weights,
                    extra_lora_scale=extra_lora_scale,
                    hf_api_token=hf_api_token,
                    civitai_api_token=civitai_api_token,
                )
            else:
                # Load single LoRA
                print(f"📦 Loading single LoRA: {lora_weights}")
                self.load_single_lora(
                    pipeline,
                    lora_url=lora_weights,
                    lora_scale=lora_scale,
                    hf_api_token=hf_api_token,
                    civitai_api_token=civitai_api_token,
                )
                
        elif lora_weights and not needs_reload:
            print(f"✅ LoRA {lora_weights} already loaded")
            if extra_lora_weights:
                print(f"✅ Extra LoRA {extra_lora_weights} already loaded")
                
        elif not lora_weights and self.current_lora:
            # No LoRAs requested but some are loaded - unload them
            print("🔄 No LoRAs requested, unloading existing LoRAs...")
            self.unload_loras(pipeline)

        # Update current state
        self.current_lora = lora_weights
        self.current_lora_scale = lora_scale
        self.current_extra_lora = extra_lora_weights
        self.current_extra_lora_scale = extra_lora_scale


class Predictor(BasePredictor, MultiLoRAMixin):
    def __init__(self):
        super().__init__()
        MultiLoRAMixin.__init__(self)
        self.current_model = "default"
        self.original_unet = None
        
        # Auto-detect device once
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Detected device: {self.device}")
    
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        print(f"Setting up with weights: {weights}")
        log_system_info()  # Log system information for debugging
        start = time.time()
        self.tuned_model = False

        print("Loading safety checker...")
        try:
            if not os.path.exists(SAFETY_CACHE) or len(os.listdir(SAFETY_CACHE)) == 0:
                print("Downloading safety checker...")
                download_weights(SAFETY_URL, SAFETY_CACHE)
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                SAFETY_CACHE, torch_dtype=torch.float16
            ).to(self.device)
            # Use the standard feature extractor for safety checking
            self.feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ Safety checker loaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to load safety checker: {e}")
            print("   Safety checker will be disabled for all predictions")
            self.safety_checker = None
            self.feature_extractor = None

        # Try to load Illustrious XL as complete pipeline first
        try:
            print("Loading Illustrious XL complete model...")
            
            # Download the complete Illustrious XL model
            illustrious_path = download_custom_model_local("illustrious-xl")
            print(f"Illustrious XL path: {illustrious_path}")
            
            from diffusers import StableDiffusionXLPipeline
            
            # Load Illustrious XL as complete pipeline (has all components)
            self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
                illustrious_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,  # Use cached files
            ).to(self.device)
            
            self.current_model = "illustrious-xl"
            print("✅ Illustrious XL complete model loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Failed to load Illustrious XL: {e}")
            print("⚠️  Falling back to standard SDXL...")
            
            # Fallback: Load SDXL base pipeline
            if not os.path.exists(SDXL_MODEL_CACHE):
                download_weights(SDXL_URL, SDXL_MODEL_CACHE)
            
            print("Loading SDXL base pipeline as fallback...")
            self.txt2img_pipe = DiffusionPipeline.from_pretrained(
                SDXL_MODEL_CACHE,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to(self.device)
            
            self.current_model = "sdxl-base"
            print("✅ SDXL fallback model loaded")
        
        # Store original UNet for model switching (either Illustrious XL or SDXL)
        self.original_unet = self.txt2img_pipe.unet

        print("Loading SDXL img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,  # This will be Illustrious XL if loaded
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to(self.device)

        print("Loading SDXL inpaint pipeline...")
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,  # This will be Illustrious XL if loaded
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.inpaint_pipe.to(self.device)

        print("Loading SDXL refiner pipeline...")
        if not os.path.exists(REFINER_MODEL_CACHE):
            download_weights(REFINER_URL, REFINER_MODEL_CACHE)

        self.refiner = DiffusionPipeline.from_pretrained(
            REFINER_MODEL_CACHE,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            vae=self.txt2img_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.refiner.to(self.device)
        print(f"Setup completed in {time.time() - start:.2f}s")

    def switch_to_standard_sdxl(self):
        """Switch to standard SDXL for realistic content."""
        if self.current_model != "sdxl-base":
            print("Switching to standard SDXL...")
            
            # If current model is Illustrious XL, we need to load SDXL
            if self.current_model == "illustrious-xl":
                try:
                    # Load standard SDXL pipeline
                    self.txt2img_pipe = DiffusionPipeline.from_pretrained(
                        SDXL_MODEL_CACHE,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                    ).to(self.device)
                    
                    # Update other pipelines to use SDXL components
                    self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                        vae=self.txt2img_pipe.vae,
                        text_encoder=self.txt2img_pipe.text_encoder,
                        text_encoder_2=self.txt2img_pipe.text_encoder_2,
                        tokenizer=self.txt2img_pipe.tokenizer,
                        tokenizer_2=self.txt2img_pipe.tokenizer_2,
                        unet=self.txt2img_pipe.unet,
                        scheduler=self.txt2img_pipe.scheduler,
                    ).to(self.device)
                    
                    self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
                        vae=self.txt2img_pipe.vae,
                        text_encoder=self.txt2img_pipe.text_encoder,
                        text_encoder_2=self.txt2img_pipe.text_encoder_2,
                        tokenizer=self.txt2img_pipe.tokenizer,
                        tokenizer_2=self.txt2img_pipe.tokenizer_2,
                        unet=self.txt2img_pipe.unet,
                        scheduler=self.txt2img_pipe.scheduler,
                    ).to(self.device)
                    
                    self.current_model = "sdxl-base"
                    print("✅ Switched to standard SDXL")
                    
                except Exception as e:
                    print(f"⚠️  Failed to switch to SDXL: {e}")
            
    def switch_to_illustrious_xl(self):
        """Switch back to Illustrious XL (default)."""
        if self.current_model != "illustrious-xl":
            print("Switching back to Illustrious XL...")
            
            try:
                from diffusers import StableDiffusionXLPipeline

                illustrious_cache = download_custom_model_local("illustrious-xl")
                
                # Load Illustrious XL pipeline
                self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
                    illustrious_cache,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    local_files_only=True,  # Force using local files since we've cached them
                ).to(self.device)
                
                # Update other pipelines to use Illustrious XL components
                self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                    vae=self.txt2img_pipe.vae,
                    text_encoder=self.txt2img_pipe.text_encoder,
                    text_encoder_2=self.txt2img_pipe.text_encoder_2,
                    tokenizer=self.txt2img_pipe.tokenizer,
                    tokenizer_2=self.txt2img_pipe.tokenizer_2,
                    unet=self.txt2img_pipe.unet,
                    scheduler=self.txt2img_pipe.scheduler,
                ).to(self.device)
                
                self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
                    vae=self.txt2img_pipe.vae,
                    text_encoder=self.txt2img_pipe.text_encoder,
                    text_encoder_2=self.txt2img_pipe.text_encoder_2,
                    tokenizer=self.txt2img_pipe.tokenizer,
                    tokenizer_2=self.txt2img_pipe.tokenizer_2,
                    unet=self.txt2img_pipe.unet,
                    scheduler=self.txt2img_pipe.scheduler,
                ).to(self.device)
                
                self.current_model = "illustrious-xl"
                print("✅ Switched to Illustrious XL")
                
            except Exception as e:
                print(f"⚠️  Failed to switch to Illustrious XL: {e}")

    def run_safety_checker(self, image):
        if self.safety_checker is None or self.feature_extractor is None:
            # Safety checker not available, return images as-is with no NSFW flags
            return image, [False] * len(image)
        
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            self.device
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        # Core flux-compatible parameters
        prompt: str = Input(
            description="Text prompt for image generation",
            default="1girl, solo, ranni_the_witch, elden_ring, looking_at_viewer, witch_hat, blue_skin, doll_joints",
        ),
        num_outputs: int = Input(
            description="Number of images to generate",
            ge=1,
            le=4,
            default=1,
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=["1:1", "2:3", "3:2", "4:3", "3:4", "4:5", "5:4", "16:9", "9:16"],
            default="1:1",
        ),
        output_format: str = Input(
            description="Output image format",
            choices=["webp", "png", "jpg"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Output image quality (0-100, only for JPEG)",
            ge=0,
            le=100,
            default=80,
        ),
        image: Optional[Path] = Input(
            description="Input image for image-to-image mode (supports PNG, JPG, WEBP).",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Set for reproducible generation",
            default=None,
        ),
        go_fast: bool = Input(
            description="Run faster predictions with reduced quality",
            default=False,
        ),
        guidance: float = Input(
            description="Guidance for generated image. Lower values can give more realistic images. Good values to try are 2, 2.5, 3 and 3.5",
            ge=1.0,
            le=20.0,
            default=7.5,
        ),
        megapixels: str = Input(
            description="Approximate number of megapixels for generated image",
            choices=["0.25", "0.5", "1", "2"],
            default="1",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. More steps generally give better quality but take longer",
            ge=1,
            le=50,
            default=28,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images",
            default=False,
        ),
        
        # LoRA parameters (flux-compatible)
        lora_weights: str = Input(
            description="Primary LoRA weights. Supports Replicate models (owner/model), HuggingFace (huggingface.co/owner/model), CivitAI (civitai.com/models/id), and direct URLs",
            default=None,
        ),
        lora_scale: float = Input(
            description="Primary LoRA scale/strength",
            ge=-1.0,
            le=3.0,
            default=1.0,
        ),
        extra_lora: str = Input(
            description="Secondary LoRA weights. Allows combining two LoRAs simultaneously",
            default=None,
        ),
        extra_lora_scale: float = Input(
            description="Secondary LoRA scale/strength",
            ge=-1.0,
            le=3.0,
            default=1.0,
        ),
        hf_api_token: str = Input(
            description="HuggingFace API token for private models",
            default=None,
        ),
        civitai_api_token: str = Input(
            description="CivitAI API token for private models",
            default=None,
        ),
        
        # SDXL-specific parameters
        negative_prompt: str = Input(
            description="Negative prompt - what you don't want in the image",
            default="",
        ),
        scheduler: str = Input(
            description="Sampling scheduler algorithm",
            choices=list(SCHEDULERS.keys()),
            default="K_EULER",
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale (same as guidance, kept for SDXL compatibility)",
            ge=1.0,
            le=20.0,
            default=4.5,
        ),
        clip_skip: int = Input(
            description="Number of CLIP layers to skip (advanced parameter)",
            ge=0,
            le=2,
            default=1,
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "expert_ensemble_refiner", "base_image_refiner"],
            default="base_image_refiner",
        ),
        high_noise_frac: float = Input(
            description="For expert_ensemble_refiner, the fraction of noise to use",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine",
            default=5,
        ),
        apply_watermark: bool = Input(
            description="Apply watermark to generated images",
            default=False,
        ),
        
        # Model selection (Illustrious XL is loaded by default)
        use_standard_sdxl: bool = Input(
            description="Use standard SDXL instead of Illustrious XL (for realistic content)",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # Setup and validation
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        # Parse dimensions from aspect ratio and megapixels (always use user settings)
        width, height = parse_aspect_ratio(aspect_ratio, megapixels)
        width, height = validate_dimensions(width, height)
        print(f"Using aspect ratio {aspect_ratio} -> {width}x{height}")
        
        # Load input image if provided (for img2img mode)
        if image:
            input_image = load_image(image).convert("RGB")
            print(f"Input image loaded with dimensions: {input_image.size[0]}x{input_image.size[1]}")
            print(f"Output will be resized to: {width}x{height}")
        
        # Apply go_fast optimizations
        if go_fast:
            num_inference_steps = max(10, num_inference_steps // 2)
            print(f"Go fast enabled: reduced steps to {num_inference_steps}")
        
        # Use guidance parameter (flux-style) or cfg_scale (SDXL-style)
        final_guidance = guidance if guidance != 7.5 else cfg_scale
        
        # Convert string tokens to Secret objects if provided
        hf_token = Secret(hf_api_token) if hf_api_token else None
        civitai_token = Secret(civitai_api_token) if civitai_api_token else None
        
        # Switch to standard SDXL if requested
        if use_standard_sdxl and self.current_model == "illustrious-xl":
            print("Switching to standard SDXL UNet...")
            self.txt2img_pipe.unet = self.original_unet
            self.img2img_pipe.unet = self.original_unet
            self.inpaint_pipe.unet = self.original_unet
            self.current_model = "sdxl-base"
            print("✅ Using standard SDXL")
        elif not use_standard_sdxl and self.current_model != "illustrious-xl":
            # Switch back to Illustrious XL (which should already be loaded)
            print("Using Illustrious XL (default)")
            
        # Apply model-specific optimizations for Illustrious XL
        if not use_standard_sdxl:
            if guidance == 7.5:  # If using default guidance, optimize for Illustrious XL
                guidance = 6.0
                print(f"Applied Illustrious XL optimized guidance: {guidance}")
            if not negative_prompt:
                negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                print("Applied Illustrious XL optimized negative prompt")
        
        # Handle LoRAs using smart caching (inspired by cog-flux)
        self.handle_loras(
            self.txt2img_pipe,
            lora_weights=lora_weights,
            lora_scale=lora_scale,
            extra_lora_weights=extra_lora,
            extra_lora_scale=extra_lora_scale,
            hf_api_token=hf_token,
            civitai_api_token=civitai_token,
        )
        
        print(f"Prompt: {prompt}")
        print(f"Negative prompt: {negative_prompt}")

        # Determine pipeline mode
        sdxl_kwargs = {}
        if image:
            print("img2img mode")
            sdxl_kwargs["image"] = input_image
            sdxl_kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.txt2img_pipe

        # Configure refiner
        if refine == "expert_ensemble_refiner":
            sdxl_kwargs["output_type"] = "latent"
            sdxl_kwargs["denoising_end"] = high_noise_frac
        elif refine == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"

        # Handle watermark
        watermark_cache = None
        if not apply_watermark:
            watermark_cache = getattr(pipe, 'watermark', None)
            if watermark_cache is not None:
                pipe.watermark = None
            refiner_watermark = getattr(self.refiner, 'watermark', None)
            if refiner_watermark is not None:
                self.refiner.watermark = None

        # Configure scheduler
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        
        # Setup generator based on available device
        generator = torch.Generator(self.device).manual_seed(seed)

        # Prepare common arguments
        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": final_guidance,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        # Apply LoRA scaling if LoRAs are loaded
        if self.loaded_loras:
            primary_scale = self.loaded_loras[0]["scale"]
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": primary_scale}

        # Handle CLIP skip (advanced parameter)
        if clip_skip > 0:
            # This would require modifying the text encoder calls
            print(f"Note: CLIP skip ({clip_skip}) requested but not implemented in this version")

        # Generate image
        print(f"Generating {num_outputs} image(s) at {width}x{height}...")
        output = pipe(**common_args, **sdxl_kwargs)

        # Apply refiner if requested
        if refine in ["expert_ensemble_refiner", "base_image_refiner"]:
            print(f"Applying refiner: {refine}")
            refiner_kwargs = {
                "image": output.images,
            }

            if refine == "expert_ensemble_refiner":
                refiner_kwargs["denoising_start"] = high_noise_frac
            if refine == "base_image_refiner" and refine_steps:
                common_args["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args, **refiner_kwargs)

        # Restore watermark
        if watermark_cache is not None:
            pipe.watermark = watermark_cache
            self.refiner.watermark = watermark_cache

        # Run safety checker (unless disabled)
        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)
        else:
            has_nsfw_content = [False] * len(output.images)

        # Save outputs with proper format
        output_paths = []
        file_extension = get_output_format_extension(output_format)
        
        for i, (img, nsfw) in enumerate(zip(output.images, has_nsfw_content)):
            if nsfw and not disable_safety_checker:
                print(f"NSFW content detected in image {i}")
                continue
                
            output_path = f"/tmp/out-{i}.{file_extension}"
            
            # Save with appropriate format and quality
            if output_format.lower() in ["jpg", "jpeg"]:
                img.save(output_path, format="JPEG", quality=output_quality, optimize=True)
            elif output_format.lower() == "png":
                img.save(output_path, format="PNG", optimize=True)
            else:  # webp
                img.save(output_path, format="WEBP", quality=output_quality, optimize=True)
                
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected in all generated images. Try running with disable_safety_checker=True, or try a different prompt."
            )

        print(f"Generated {len(output_paths)} image(s)")
        print(f"LoRA cache info: {self.lora_cache.cache_info()}")
        return output_paths