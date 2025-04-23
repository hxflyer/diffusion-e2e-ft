# Texture Map Generation with Diffusion Models

This project extends the diffusion-e2e-ft framework to support generating texture maps from head images. It allows you to train a model that takes a head image as input and generates a corresponding texture map.

## Setup

1. Follow the main project setup instructions in the README.md file.
2. Prepare your dataset in the `data/head_texture` directory:
   - Place head images in `data/head_texture/heads/`
   - Place corresponding texture maps in `data/head_texture/textures/`
   - Make sure each head image has a matching texture map with the same filename

## Training

To train a texture map generation model:

```bash
# Make the script executable
chmod +x training/scripts/train_texture_map_generator.sh

# Run the training script
./training/scripts/train_texture_map_generator.sh
```

The script will train a model using Stable Diffusion v1.5 as the base model. The trained model will be saved to `model-finetuned/texture_map_generator`.

## Inference

To generate texture maps from head images using a trained model:

```bash
python inference_texture.py \
  --checkpoint model-finetuned/texture_map_generator \
  --input_dir path/to/head/images \
  --output_dir path/to/output/textures
```

### Inference Options

- `--checkpoint`: Path to the trained model checkpoint (default: model-finetuned/texture_map_generator)
- `--input_dir`: Directory containing input head images
- `--output_dir`: Directory to save generated texture maps
- `--device`: Device to run inference on (cuda/cpu)
- `--batch_size`: Batch size for inference (default: 1)
- `--image_size`: Size to resize input images to (default: 512)
- `--half_precision`: Use half precision (float16) for inference

## How It Works

The system uses a fine-tuned diffusion model to generate texture maps from head images:

1. **Training**: The model is trained end-to-end using a combination of L1 loss and perceptual loss to learn the mapping from head images to texture maps.

2. **Inference**: During inference, the model takes a head image as input and generates a corresponding texture map in a single step.

3. **Pipeline**: The custom `TextureMapPipeline` class handles the inference process, including image preprocessing, model execution, and postprocessing.

## Customization

You can customize the training process by modifying:

- `training/scripts/train_texture_map_generator.sh`: Change training parameters
- `training/util/loss.py`: Modify the `TextureLoss` class to use different loss functions
- `texture_pipeline.py`: Customize the inference pipeline

## Requirements

- PyTorch
- Diffusers
- Transformers
- Accelerate
- PIL
- NumPy
