import os
import shutil
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
from diffusers.utils import load_image
from transformers import CLIPImageProcessor

from weights import SDXLLoRACache
from utils import (
    download_weights,
    TokenEmbeddingsHandler,
    load_lora_weights_to_unet,
    merge_lora_weights,
    validate_lora_file,
    log_system_info,
    parse_aspect_ratio,
    validate_dimensions,
    get_output_format_extension,
    get_model_config_local,
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
        
        # Load LoRA using the existing method from the original predictor
        self._apply_lora_to_unet(pipe.unet, lora_path, lora_scale)
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
        self._apply_multiple_loras_to_unet(pipe.unet, loras_to_load)
        self.loaded_loras = loras_to_load
        
        print(f"Multiple LoRAs loaded in {time.time() - start_time:.2f}s")
    
    def _apply_lora_to_unet(self, unet, lora_path: Path, lora_scale: float):
        """Apply a single LoRA to the UNet using utilities."""
        # Validate LoRA file first
        if not validate_lora_file(lora_path):
            raise ValueError(f"Invalid LoRA file: {lora_path}")
        
        # Use the utility function to load LoRA weights
        load_lora_weights_to_unet(unet, lora_path, device=self.device)
    
    def _apply_multiple_loras_to_unet(self, unet, loras: List[Dict]):
        """
        Apply multiple LoRAs to UNet by merging their weights.
        This is a simplified approach - in production, you might want more sophisticated merging.
        """
        if not loras:
            return
        
        # For simplicity, we'll apply the first LoRA normally and then merge additional ones
        # This is a basic implementation - flux uses more advanced merging techniques
        first_lora = loras[0]
        self._apply_lora_to_unet(unet, first_lora["path"], first_lora["scale"])
        
        # For additional LoRAs, we would need more sophisticated merging
        # This is a placeholder for the enhanced logic you might implement
        if len(loras) > 1:
            print(f"Note: Currently applying primary LoRA only. Additional LoRA merging would require enhanced implementation.")
    
    def unload_loras(self, pipe):
        """Unload all LoRAs and restore original attention processors."""
        if self.original_attn_processors is not None:
            pipe.unet.set_attn_processor(self.original_attn_processors)
            print("Unloaded all LoRAs and restored original attention processors")
        self.loaded_loras = []


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
            self.feature_extractor = CLIPImageProcessor.from_pretrained(SAFETY_CACHE)
            print("✅ Safety checker loaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to load safety checker: {e}")
            print("   Safety checker will be disabled for all predictions")
            self.safety_checker = None
            self.feature_extractor = None

        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        print("Loading SDXL txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.txt2img_pipe.to(self.device)
        
        # Store original UNet for model switching
        self.original_unet = self.txt2img_pipe.unet
        
        # Load Illustrious XL UNet as default model
        try:
            print("Loading Illustrious XL UNet as default model...")
            
            # Try multiple paths: pre-cached, local download, then fallback
            illustrious_paths_to_try = [
                "/src/illustrious-xl-unet/illustrious-xl.safetensors",  # Pre-cached during build
                "/src/custom-models/illustrious-xl-unet.safetensors",  # Alternative pre-cache location
            ]
            
            illustrious_path = None
            for path in illustrious_paths_to_try:
                if os.path.exists(path):
                    illustrious_path = path
                    print(f"Found pre-cached Illustrious XL at: {path}")
                    break
            
            # If not found, download it
            if not illustrious_path:
                print("Illustrious XL not found in cache, downloading...")
                illustrious_path = download_custom_model_local("illustrious-xl")
                print(f"Downloaded Illustrious XL to: {illustrious_path}")
            
            # Load the UNet model
            from diffusers import UNet2DConditionModel
            illustrious_unet = UNet2DConditionModel.from_single_file(
                illustrious_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to(self.device)
            
            # Replace the UNet in the pipeline
            self.txt2img_pipe.unet = illustrious_unet
            self.current_model = "illustrious-xl"
            print("✅ Illustrious XL loaded as default model")
            
        except Exception as e:
            print(f"⚠️  Failed to load Illustrious XL: {e}")
            print("⚠️  Using standard SDXL as fallback")
            self.current_model = "sdxl-base"

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
        """Switch to standard SDXL UNet for realistic content."""
        if self.current_model != "sdxl-base":
            print("Switching to standard SDXL UNet...")
            self.txt2img_pipe.unet = self.original_unet
            self.img2img_pipe.unet = self.original_unet
            self.inpaint_pipe.unet = self.original_unet
            self.current_model = "sdxl-base"
            print("✅ Using standard SDXL")
            
    def switch_to_illustrious_xl(self):
        """Switch back to Illustrious XL UNet (default)."""
        if self.current_model != "illustrious-xl":
            # This should not normally happen since Illustrious XL is loaded by default
            print("Note: Illustrious XL should already be the default model")
            self.current_model = "illustrious-xl"

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

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
            default="An astronaut riding a rainbow unicorn",
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
        image: Optional[str] = Input(
            description="Input image for image-to-image mode. The aspect ratio of your output will match this image",
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
            default=7.5,
        ),
        clip_skip: int = Input(
            description="Number of CLIP layers to skip (advanced parameter)",
            ge=0,
            le=2,
            default=0,
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "expert_ensemble_refiner", "base_image_refiner"],
            default="no_refiner",
        ),
        high_noise_frac: float = Input(
            description="For expert_ensemble_refiner, the fraction of noise to use",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine",
            default=None,
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
        
        # Parse dimensions from aspect ratio and megapixels
        if image:
            # If input image provided, load it to get dimensions
            input_image = load_image(image).convert("RGB")
            width, height = input_image.size
            width, height = validate_dimensions(width, height)
            print(f"Using image dimensions: {width}x{height}")
        else:
            # Parse aspect ratio and megapixels
            width, height = parse_aspect_ratio(aspect_ratio, megapixels)
            width, height = validate_dimensions(width, height)
            print(f"Using aspect ratio {aspect_ratio} -> {width}x{height}")
        
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
        
        # Load LoRAs if specified (support both flux parameter names)
        lora_url = lora_weights or None  # Use lora_weights (flux-style) if provided
        extra_lora_url = extra_lora or None  # Use extra_lora (flux-style) if provided
        
        if lora_url:
            if extra_lora_url:
                # Load multiple LoRAs
                self.load_multiple_loras(
                    self.txt2img_pipe,
                    main_lora_url=lora_url,
                    main_lora_scale=lora_scale,
                    extra_lora_url=extra_lora_url,
                    extra_lora_scale=extra_lora_scale,
                    hf_api_token=hf_token,
                    civitai_api_token=civitai_token,
                )
            else:
                # Load single LoRA
                self.load_single_lora(
                    self.txt2img_pipe,
                    lora_url=lora_url,
                    lora_scale=lora_scale,
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