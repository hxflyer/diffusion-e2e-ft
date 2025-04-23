#!/usr/bin/env python
# Script to test the HeadTextureDataset class with the generated test data

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from training.dataloaders.load import HeadTextureDataset

def main():
    print("Testing HeadTextureDataset with generated test data...")
    
    # Create dataset
    dataset = HeadTextureDataset(root_dir="data/head_texture", transform=True, image_size=(512, 512))
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)} image pairs")
    
    # Test loading a few samples
    num_samples = min(5, len(dataset))
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    
    for i in range(num_samples):
        # Get a sample
        sample = dataset[i]
        
        # Convert tensors to PIL images for display
        head_img = tensor_to_pil(sample["rgb"])
        texture_img = tensor_to_pil(sample["texture"])
        
        # Display images
        axes[i, 0].imshow(head_img)
        axes[i, 0].set_title(f"Head Image {i+1}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(texture_img)
        axes[i, 1].set_title(f"Texture Map {i+1}")
        axes[i, 1].axis("off")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("dataset_test_samples.png")
    print("Test samples saved to dataset_test_samples.png")
    
    # Test batch loading
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(dataloader))
    
    print(f"Batch shapes:")
    print(f"RGB (head images): {batch['rgb'].shape}")
    print(f"Depth (texture for VAE): {batch['depth'].shape}")
    print(f"Texture (target): {batch['texture'].shape}")
    print(f"Validity mask: {batch['val_mask'].shape}")
    print(f"Domain: {batch['domain']}")
    
    print("Dataset test completed successfully!")

def tensor_to_pil(tensor):
    """Convert a normalized tensor to a PIL image"""
    # Convert from [-1, 1] to [0, 1]
    img = (tensor + 1.0) / 2.0
    
    # Convert to numpy and transpose
    img = img.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Clip values to [0, 1]
    img = img.clip(0, 1)
    
    # Convert to uint8
    img = (img * 255).astype('uint8')
    
    return img

if __name__ == "__main__":
    main()
