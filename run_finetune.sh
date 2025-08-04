#!/usr/bin/env bash
# Run Gemma 3 27B fine-tuning with optimal settings
set -euo pipefail

# Default options (can be overridden by environment variables)
DATA_PATH=${DATA_PATH:-"/home/models/FAISS_INGEST/scraped_data.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./gemma-3-finetuned"}
NUM_GPUS=${NUM_GPUS:-4}  # Default to 4 GPUs
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}  # Per device batch size
GRAD_ACCUM=${GRAD_ACCUM:-4}  # Gradient accumulation steps
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
SAVE_STEPS=${SAVE_STEPS:-100}
LR=${LR:-2e-5}

# Required packages (install manually with pip):
# - torch (compatible with your CUDA version)
# - transformers>=4.36.0
# - peft>=0.7.0
# - accelerate>=0.25.0
# - bitsandbytes>=0.41.0
# - datasets>=2.14.0
# - sentencepiece>=0.1.99
# - protobuf>=4.25.0

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print setup information
echo "==========================================="
echo "Gemma 3 27B Fine-tuning"
echo "==========================================="
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Using $NUM_GPUS GPUs"
echo "Training for $EPOCHS epochs"
echo "Batch size: $BATCH_SIZE per device"
echo "Gradient accumulation: $GRAD_ACCUM steps"
echo "Learning rate: $LR"
echo "LoRA rank: $LORA_RANK"
echo "LoRA alpha: $LORA_ALPHA"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "==========================================="

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export TORCH_COMPILE_MODE=reduce-overhead
export TORCH_INDUCTOR_COMPILE_TO_EAGER=1
export TORCH_CUDNN_V8_API_DISABLED=1

# Set large shared memory size for DataLoader workers
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Launch training with accelerate for distributed training
accelerate launch --multi_gpu --num_processes=$NUM_GPUS finetune_gemma.py \
    --model_name_or_path="google/gemma-3-27b-it" \
    --data_path="$DATA_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --num_train_epochs="$EPOCHS" \
    --per_device_train_batch_size="$BATCH_SIZE" \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --learning_rate="$LR" \
    --lora_r="$LORA_RANK" \
    --lora_alpha="$LORA_ALPHA" \
    --max_seq_length="$MAX_SEQ_LENGTH" \
    --save_steps="$SAVE_STEPS" \
    --bf16 \
    --tf32 \
    --gradient_checkpointing

echo "Fine-tuning complete! Model saved to $OUTPUT_DIR"
echo "To use your fine-tuned model, run:"
echo "python -c \"from peft import PeftModel; from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('google/gemma-3-27b-it', device_map='auto'); model = PeftModel.from_pretrained(model, '$OUTPUT_DIR'); tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-27b-it')\"" 