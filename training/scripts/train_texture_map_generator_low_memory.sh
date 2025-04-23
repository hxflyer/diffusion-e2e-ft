#!/bin/bash 

# Set PyTorch memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# Run training with extreme memory optimizations using CPU offloading
accelerate launch --config_file training/scripts/low_memory_config.yaml training/train.py \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --modality "texture" \
  --noise_type "zeros" \
  --max_train_steps 20000 \
  --checkpointing_steps 2000 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --gradient_checkpointing \
  --learning_rate 3e-05 \
  --lr_total_iter_length 20000 \
  --lr_exp_warmup_steps 100 \
  --mixed_precision "fp16" \
  --output_dir "model-finetuned/texture_map_generator" \
  --enable_xformers_memory_efficient_attention \
  "$@"
