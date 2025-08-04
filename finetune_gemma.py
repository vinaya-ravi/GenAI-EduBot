#!/usr/bin/env python3
# Gemma 3 27B Fine-tuning Script using LoRA
# Usage: python finetune_gemma.py --data_path your_data.json --output_dir ./output

import os
import json
import torch
import argparse
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import bitsandbytes as bnb
from accelerate import Accelerator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments for the model configuration"""
    model_name_or_path: str = field(
        default="google/gemma-3-27b-it",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization to reduce memory usage"}
    )
    use_nested_quant: bool = field(
        default=True,
        metadata={"help": "Use nested quantization for 4-bit (doubles the compression)"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit quantization (fp4 or nf4)"}
    )

@dataclass
class DataArguments:
    """Arguments for data processing"""
    data_path: str = field(
        default="./training_data.json",
        metadata={"help": "Path to the training data JSON file"}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length to use for training"}
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable streaming mode for large datasets"}
    )

@dataclass
class LoraArguments:
    """Arguments for LoRA fine-tuning"""
    lora_r: int = field(
        default=16,
        metadata={"help": "Lora rank - lower rank for memory efficiency, higher for capacity"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Lora alpha - scaling factor"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Lora dropout rate"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Modules to apply LoRA to"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias configuration for LoRA (none, all, or lora_only)"}
    )

def create_sample_data(output_file="sample_training_data.json"):
    """Create a sample training data file if one doesn't exist"""
    if os.path.exists(output_file):
        logger.info(f"Sample data file {output_file} already exists. Skipping creation.")
        return
    
    sample_data = [
        {
            "messages": [
                {"role": "user", "content": "Write a professional email to request an extension on a project deadline"},
                {"role": "assistant", "content": "Subject: Request for Project Deadline Extension\n\nDear [Recipient],\n\nI hope this email finds you well. I am writing to request an extension for the [Project Name] deadline, currently set for [Current Date].\n\nDue to [specific reason for extension], I am experiencing some delays in completing the project to the high standard required. I am requesting an extension until [Proposed New Date], which would allow me to deliver a thorough and well-executed project.\n\nI understand the importance of meeting deadlines and apologize for any inconvenience this may cause. I am committed to delivering quality work and appreciate your consideration.\n\nThank you for your understanding.\n\nSincerely,\n[Your Name]\n[Your Contact Information]"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain quantum computing in simple terms"},
                {"role": "assistant", "content": "Quantum computing is like having a super-powered calculator that can consider many possibilities at once, unlike regular computers that look at one thing at a time.\n\nRegular computers use bits (0s and 1s) to process information. Quantum computers use quantum bits or 'qubits' that can be 0, 1, or both at the same time - a state called 'superposition.'\n\nImagine trying to find your way through a maze. A regular computer would try one path at a time. A quantum computer could explore multiple paths simultaneously, making it potentially much faster for certain problems.\n\nThis technology is still developing but could eventually revolutionize fields like medicine (designing new drugs), cryptography (creating unbreakable codes), and complex simulations (like weather forecasting)."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How can I improve my study habits?"},
                {"role": "assistant", "content": "Here are some effective ways to improve your study habits:\n\n1. Create a dedicated study space free from distractions\n2. Develop a consistent schedule with regular study times\n3. Break study sessions into 25-30 minute focused intervals (Pomodoro Technique)\n4. Use active learning strategies like summarizing information in your own words\n5. Practice retrieval by testing yourself rather than just re-reading\n6. Teach concepts to someone else to solidify your understanding\n7. Get enough sleep - your brain consolidates memories during rest\n8. Stay hydrated and maintain proper nutrition\n9. Use spaced repetition to review material at increasing intervals\n10. Set specific, achievable goals for each study session\n\nRemember that everyone learns differently, so experiment with these techniques to find what works best for you."}
            ]
        }
    ]
    
    with open(output_file, "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Created sample training data file: {output_file}")

def format_chat_for_training(example):
    """Format chat conversations into the required format for fine-tuning"""
    
    if "messages" in example:
        # ChatML format
        messages = example["messages"]
        formatted_text = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_text += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_text += f"<|assistant|>\n{content}\n"
        
        formatted_text += "<|end|>"
        return {"text": formatted_text}
    
    elif "prompt" in example and "completion" in example:
        # Simple prompt-completion pairs
        prompt = example["prompt"]
        completion = example["completion"]
        return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{completion}\n<|end|>"}
    
    elif "question" in example and "answer" in example:
        # QA format
        question = example["question"]
        answer = example["answer"]
        return {"text": f"<|user|>\n{question}\n<|assistant|>\n{answer}\n<|end|>"}
    
    elif "input" in example and "output" in example:
        # Input-output format
        input_text = example["input"]
        output_text = example["output"]
        return {"text": f"<|user|>\n{input_text}\n<|assistant|>\n{output_text}\n<|end|>"}
        
    elif "text" in example:
        # Already formatted text
        return {"text": example["text"]}
    
    else:
        # If the format doesn't match any known structure, return empty
        logger.warning(f"Unrecognized example format: {example.keys()}")
        return {"text": ""}

def measure_processing_speed(model, tokenizer, dataset, batch_size=1, num_batches=5):
    """Measure the processing speed of the model on a few batches"""
    logger.info("Measuring processing speed...")
    start_time = time.time()
    
    # Take a small subset of the dataset
    test_dataset = dataset.select(range(min(100, len(dataset))))
    
    # Create a small dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Process a few batches
    for i, batch in enumerate(test_dataloader):
        if i >= num_batches:
            break
            
        # Process the batch
        inputs = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        if i % 2 == 0:
            logger.info(f"Processed batch {i+1}/{num_batches}")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_batches
    samples_per_second = (batch_size * num_batches) / total_time
    
    logger.info(f"Average time per batch: {avg_time_per_batch:.2f} seconds")
    logger.info(f"Processing speed: {samples_per_second:.2f} samples/second")
    
    return samples_per_second

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    
    # Handle both command line and JSON file for arguments
    if len(os.sys.argv) == 2 and os.sys.argv[1].endswith(".json"):
        model_args, data_args, lora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(os.sys.argv[1])
        )
    else:
        # Add default training arguments
        default_args = [
            "--output_dir", "./gemma-3-finetuned",
            "--num_train_epochs", "2",
            "--per_device_train_batch_size", "4",
            "--gradient_accumulation_steps", "2",
            "--save_strategy", "steps",
            "--save_steps", "100",
            "--save_total_limit", "3",
            "--learning_rate", "2e-5",
            "--lr_scheduler_type", "cosine",
            "--warmup_steps", "100",
            "--logging_steps", "10",
            "--bf16", "True",
            "--tf32", "True",
            "--gradient_checkpointing", "True",
            "--report_to", "none"
        ]
        
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses(
            args=default_args if len(os.sys.argv) == 1 else None
        )
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Check if data file exists, otherwise create sample data
    if not os.path.exists(data_args.data_path):
        logger.info(f"Data file {data_args.data_path} not found. Creating sample data.")
        create_sample_data(data_args.data_path)
    
    # Load the model and tokenizer
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # Configure 4-bit quantization if requested
    compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
    bnb_config = None
    
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )
    
    # Load the model with quantization if configured
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,  # Fast tokenizers may not work properly with Gemma
    )
    
    # Add special tokens if they don't exist
    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    special_tokens_dict = {}
    
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<|padding|>"
    
    # Add chat special tokens if they don't exist
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            if "additional_special_tokens" not in special_tokens_dict:
                special_tokens_dict["additional_special_tokens"] = []
            special_tokens_dict["additional_special_tokens"].append(token)
    
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    
    # Prepare the model for 4-bit training if using quantization
    if model_args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info(f"Configuring LoRA with rank {lora_args.lora_r} and alpha {lora_args.lora_alpha}")
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_args.target_modules,
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and preprocess dataset
    logger.info(f"Loading dataset from {data_args.data_path}")
    dataset = load_dataset(
        "json", 
        data_files=data_args.data_path, 
        streaming=data_args.streaming,
        split="train"
    )
    
    # Format the data for training
    logger.info("Preprocessing dataset")
    dataset = dataset.map(format_chat_for_training)
    
    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # After loading the dataset and before training
    logger.info("Running speed test...")
    samples_per_second = measure_processing_speed(model, tokenizer, tokenized_dataset)
    
    # Calculate estimated training time
    total_samples = len(tokenized_dataset)
    total_batches = total_samples / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    estimated_time_per_epoch = total_batches / samples_per_second
    total_estimated_time = estimated_time_per_epoch * training_args.num_train_epochs
    
    logger.info(f"Estimated time per epoch: {estimated_time_per_epoch/3600:.2f} hours")
    logger.info(f"Total estimated training time: {total_estimated_time/3600:.2f} hours")
    
    # Ask user if they want to proceed with training
    proceed = input("Do you want to proceed with training? (y/n): ")
    if proceed.lower() != 'y':
        logger.info("Training cancelled by user")
        return
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    from transformers.utils import logging as hf_logging
    from transformers import BitsAndBytesConfig
    
    # Set verbose logging for debugging
    hf_logging.set_verbosity_info()
    
    # Add ability to handle CTRL+C gracefully
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving model...") 