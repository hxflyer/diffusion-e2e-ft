#!/usr/bin/env python
# Inference script for texture map generation

import os
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from texture_pipeline import TextureMapPipeline
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Generate texture maps from head images")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model-finetuned/texture_map_generator",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input head images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated texture maps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size to resize input images to (square)",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Use half precision (float16) for inference",
    )
    return parser.parse_args()

def load_images(input_dir, image_size):
    """Load all images from the input directory"""
    images = []
    filenames = []
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)  # Scale to [-1, 1]
    ])
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(transform(image))
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    
    return images, filenames

def process_batch(model, batch_images, device):
    """Process a batch of images through the model"""
    # Convert list of tensors to a batch
    batch_tensor = torch.stack(batch_images).to(device)
    
    # Generate texture maps
    with torch.no_grad():
        output = model(batch_tensor)
    
    return output.images

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    dtype = torch.float16 if args.half_precision else torch.float32
    
    # Load the components from the checkpoint
    components = StableDiffusionPipeline.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
    )
    
    # Create our custom pipeline
    pipeline = TextureMapPipeline(
        vae=components.vae,
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        unet=components.unet,
        scheduler=components.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    ).to(args.device)
    
    # Set to deterministic mode for single-step inference
    pipeline.scheduler.config.timestep_spacing = "trailing"
    pipeline.scheduler.timesteps = torch.tensor([999], device=args.device)
    
    # Load images
    print(f"Loading images from {args.input_dir}")
    images, filenames = load_images(args.input_dir, args.image_size)
    
    if not images:
        print("No valid images found in the input directory")
        return
    
    print(f"Processing {len(images)} images")
    
    # Process images in batches
    for i in range(0, len(images), args.batch_size):
        batch_images = images[i:i+args.batch_size]
        batch_filenames = filenames[i:i+args.batch_size]
        
        # Generate texture maps
        outputs = pipeline(
            batch_images,
            num_inference_steps=1,
            guidance_scale=1.0,
        ).images
        
        # Save outputs
        for output, filename in zip(outputs, batch_filenames):
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(args.output_dir, f"{base_name}_texture.png")
            output.save(output_path)
            print(f"Saved texture map to {output_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()
