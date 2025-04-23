#!/usr/bin/env python
# Custom pipeline for texture map generation

import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import numpy as np

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging

logger = logging.get_logger(__name__)

class TextureMapPipeline(StableDiffusionPipeline):
    """
    Pipeline for generating texture maps from head images using a fine-tuned diffusion model.
    """
    
    def __call__(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]],
        num_inference_steps: int = 1,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[callable] = None,
        callback_steps: int = 1,
        **kwargs,
    ) -> Union[StableDiffusionPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for inference.
        
        Args:
            image: The input image(s) to generate texture maps for.
            num_inference_steps: The number of denoising steps (should be 1 for E2E fine-tuned models).
            guidance_scale: Guidance scale for classifier-free guidance (should be 1.0 for E2E fine-tuned models).
            negative_prompt: Ignored for this pipeline.
            generator: Random number generator for reproducibility.
            latents: Pre-generated latent noise samples.
            output_type: The output format of the generated image (pil or tensor).
            return_dict: Whether to return a StableDiffusionPipelineOutput or tuple.
            callback: A function that will be called every `callback_steps` steps.
            callback_steps: Number of steps between callbacks.
            
        Returns:
            StableDiffusionPipelineOutput or tuple: The generated texture maps.
        """
        # 0. Convert input image to tensor if it's a PIL image
        if isinstance(image, (PIL.Image.Image, list)):
            if isinstance(image, list):
                image = [self.preprocess_image(img) for img in image]
                image = torch.cat(image, dim=0)
            else:
                image = self.preprocess_image(image).unsqueeze(0)
        
        # 1. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # 2. Prepare latent variables
        batch_size = image.shape[0]
        
        # 3. Encode input image
        image_latents = self.encode_image(image)
        
        # 4. Prepare noise latents
        noise_latents = torch.zeros_like(image_latents).to(self.device)
        
        # 5. Prepare text embeddings (empty for this pipeline)
        text_embeddings = self._get_empty_text_embeddings(batch_size)
        
        # 6. Prepare model input
        if self.unet.config.in_channels == 8:
            # For models with doubled input channels (RGB + noise)
            model_input = torch.cat([image_latents, noise_latents], dim=1)
        else:
            # For models with standard input channels (just RGB)
            model_input = image_latents
        
        # 7. Generate texture map
        model_output = self.unet(
            model_input,
            timesteps[0],
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]
        
        # 8. Convert model output to latent
        alpha_prod_t = self.scheduler.alphas_cumprod[timesteps[0]].to(self.device)
        beta_prod_t = 1 - alpha_prod_t
        
        if self.scheduler.config.prediction_type == "v_prediction":
            latent_output = (alpha_prod_t**0.5) * noise_latents - (beta_prod_t**0.5) * model_output
        elif self.scheduler.config.prediction_type == "epsilon":
            latent_output = (noise_latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "sample":
            latent_output = model_output
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        
        # 9. Decode latent to image
        texture_maps = self.decode_latents(latent_output)
        
        # 10. Convert to output format
        if output_type == "pil":
            texture_maps = self.numpy_to_pil(texture_maps)
        
        if not return_dict:
            return (texture_maps,)
        
        return StableDiffusionPipelineOutput(images=texture_maps)
    
    def preprocess_image(self, image):
        """
        Preprocess an input image for the pipeline.
        """
        if isinstance(image, torch.Tensor):
            return image
        
        if isinstance(image, PIL.Image.Image):
            image = image.convert("RGB")
            image = np.array(image)
            image = image.astype(np.float32) / 255.0
            image = image * 2.0 - 1.0  # Scale to [-1, 1]
            image = image.transpose(2, 0, 1)  # Convert to channel-first format
            image = torch.from_numpy(image)
            return image.to(self.device)
    
    def encode_image(self, image):
        """
        Encode an image to latent space using the VAE.
        """
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent
    
    def decode_latents(self, latents):
        """
        Decode latents to images using the VAE.
        """
        latents = 1 / self.vae.config.scaling_factor * latents
        
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        
        return images
    
    def _get_empty_text_embeddings(self, batch_size):
        """
        Get empty text embeddings for the model.
        """
        # Tokenize empty string
        inputs = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        inputs = inputs.input_ids.to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(inputs)[0]
        
        # Expand to batch size
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
        
        return text_embeddings
