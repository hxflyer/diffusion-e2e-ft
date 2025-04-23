#!/usr/bin/env python
# Script to generate test data for head-to-texture training

import os
import numpy as np
from PIL import Image, ImageDraw
import random
import math

# Ensure directories exist
os.makedirs("data/head_texture/heads", exist_ok=True)
os.makedirs("data/head_texture/textures", exist_ok=True)

# Configuration
NUM_IMAGES = 100
IMAGE_SIZE = 512
TEXTURE_SIZE = 512
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 128, 255),  # Light Blue
    (128, 255, 0),  # Lime
]

def generate_head_image(index, size=IMAGE_SIZE):
    """Generate a synthetic head image"""
    # Create a blank image with a random background color
    bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    image = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Draw a head-like ellipse
    head_color = (random.randint(150, 200), random.randint(100, 150), random.randint(50, 100))
    padding = size // 8
    draw.ellipse(
        [(padding, padding), (size - padding, size - padding)],
        fill=head_color
    )
    
    # Add some facial features
    # Eyes
    eye_size = size // 12
    eye_y = size // 3
    eye_spacing = size // 4
    eye_color = (255, 255, 255)
    pupil_color = (0, 0, 0)
    
    # Left eye
    left_eye_x = size // 2 - eye_spacing // 2
    draw.ellipse(
        [(left_eye_x - eye_size, eye_y - eye_size), 
         (left_eye_x + eye_size, eye_y + eye_size)],
        fill=eye_color
    )
    draw.ellipse(
        [(left_eye_x - eye_size//2, eye_y - eye_size//2), 
         (left_eye_x + eye_size//2, eye_y + eye_size//2)],
        fill=pupil_color
    )
    
    # Right eye
    right_eye_x = size // 2 + eye_spacing // 2
    draw.ellipse(
        [(right_eye_x - eye_size, eye_y - eye_size), 
         (right_eye_x + eye_size, eye_y + eye_size)],
        fill=eye_color
    )
    draw.ellipse(
        [(right_eye_x - eye_size//2, eye_y - eye_size//2), 
         (right_eye_x + eye_size//2, eye_y + eye_size//2)],
        fill=pupil_color
    )
    
    # Nose
    nose_width = size // 16
    nose_height = size // 8
    nose_y = size // 2
    nose_x = size // 2
    draw.polygon(
        [(nose_x, nose_y - nose_height), 
         (nose_x - nose_width, nose_y + nose_height), 
         (nose_x + nose_width, nose_y + nose_height)],
        fill=(100, 50, 50)
    )
    
    # Mouth
    mouth_width = size // 4
    mouth_height = size // 16
    mouth_y = size * 2 // 3
    draw.ellipse(
        [(size//2 - mouth_width, mouth_y - mouth_height), 
         (size//2 + mouth_width, mouth_y + mouth_height)],
        fill=(200, 0, 0)
    )
    
    # Add some random patterns to make each head unique
    for _ in range(5):
        x = random.randint(padding, size - padding)
        y = random.randint(padding, size - padding)
        radius = random.randint(5, 20)
        color = random.choice(COLORS)
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=color)
    
    # Add index number to the image for easy identification
    draw.text((10, 10), f"Head {index}", fill=(0, 0, 0))
    
    return image

def generate_texture_map(index, head_image, size=TEXTURE_SIZE):
    """Generate a corresponding texture map for a head image"""
    # Create a new image for the texture
    texture = Image.new('RGB', (size, size), (200, 200, 200))
    draw = ImageDraw.Draw(texture)
    
    # Get the head image data
    head_data = np.array(head_image)
    
    # Create a UV unwrapping-like pattern
    # Divide the texture into a grid
    grid_size = 8
    cell_size = size // grid_size
    
    # Fill each grid cell with a color derived from the head image
    for i in range(grid_size):
        for j in range(grid_size):
            # Sample a point from the head image
            sample_x = int((i / grid_size) * head_image.width)
            sample_y = int((j / grid_size) * head_image.height)
            
            # Get the color at that point
            if 0 <= sample_x < head_data.shape[1] and 0 <= sample_y < head_data.shape[0]:
                color = tuple(head_data[sample_y, sample_x])
            else:
                color = (128, 128, 128)
            
            # Draw a rectangle in the texture map
            draw.rectangle(
                [(i * cell_size, j * cell_size), 
                 ((i + 1) * cell_size, (j + 1) * cell_size)],
                fill=color
            )
    
    # Add some patterns to make it look like a texture map
    # Draw seam lines
    for i in range(1, grid_size):
        draw.line([(i * cell_size, 0), (i * cell_size, size)], fill=(0, 0, 0), width=2)
        draw.line([(0, i * cell_size), (size, i * cell_size)], fill=(0, 0, 0), width=2)
    
    # Add some UV coordinate-like markings
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            x = i * cell_size
            y = j * cell_size
            draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=(255, 0, 0))
    
    # Add some random patterns unique to this texture
    for _ in range(10):
        x = random.randint(0, size)
        y = random.randint(0, size)
        radius = random.randint(5, 15)
        color = random.choice(COLORS)
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=color)
    
    # Add index number to the texture for easy identification
    draw.text((10, 10), f"Texture {index}", fill=(0, 0, 0))
    
    return texture

def main():
    print(f"Generating {NUM_IMAGES} test image pairs...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Ensure directories exist
    head_dir = "data/head_texture/heads"
    texture_dir = "data/head_texture/textures"
    
    os.makedirs(head_dir, exist_ok=True)
    os.makedirs(texture_dir, exist_ok=True)
    
    print(f"Head directory: {head_dir}")
    print(f"Texture directory: {texture_dir}")
    
    try:
        for i in range(1, NUM_IMAGES + 1):
            try:
                # Generate head image
                head_image = generate_head_image(i)
                head_path = f"{head_dir}/head_{i:03d}.png"
                head_image.save(head_path)
                
                # Generate corresponding texture map
                texture_image = generate_texture_map(i, head_image)
                texture_path = f"{texture_dir}/head_{i:03d}.png"
                texture_image.save(texture_path)
                
                print(f"Generated image pair {i}: {head_path} and {texture_path}")
            except Exception as e:
                print(f"Error generating image pair {i}: {e}")
        
        # Count the generated files
        head_count = len([f for f in os.listdir(head_dir) if f.endswith('.png')])
        texture_count = len([f for f in os.listdir(texture_dir) if f.endswith('.png')])
        
        print(f"Successfully generated {head_count} head images and {texture_count} texture maps:")
        print(f"- Head images: {head_dir}/")
        print(f"- Texture maps: {texture_dir}/")
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
