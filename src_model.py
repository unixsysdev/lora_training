"""
Model loading and inference utilities.
"""
import os
from typing import Dict, Tuple, Any

import torch
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from dotenv import load_dotenv

from src_logger import log_event

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = os.getenv("MODEL")
QUANT_BITS = int(os.getenv("QUANT_BITS", "4"))
DEVICE = os.getenv("TORCH_DEVICE", "cuda")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "models/adapter.safetensors")

# Compute types
COMPUTE_DTYPE = os.getenv("TORCH_COMPUTE_DTYPE", "bfloat16")
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32
}
COMPUTE_DTYPE_TORCH = DTYPE_MAP.get(COMPUTE_DTYPE, torch.bfloat16)

# LORA parameters
LORA_RANK = int(os.getenv("LORA_RANK", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
TARGET_MODULES = os.getenv("LORA_MODULES", "q_proj,v_proj").split(",")

# Global model and tokenizer
_model = None
_tokenizer = None

def load_model() -> AutoModelForCausalLM:
    """
    Load the model with appropriate quantization settings.
    
    Returns:
        The loaded model with LoRA configuration
    """
    log_event("model_loading_started", model=MODEL_NAME, quant_bits=QUANT_BITS)
    
    quantization_config = None
    model_kwargs = {"device_map": "auto" if DEVICE == "cuda" else DEVICE}
    
    # Configure quantization
    if QUANT_BITS == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE_TORCH,
            bnb_4bit_use_double_quant=True
        )
    elif QUANT_BITS == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=COMPUTE_DTYPE_TORCH
        )
    else:  # 16-bit
        model_kwargs["torch_dtype"] = COMPUTE_DTYPE_TORCH
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs
    )
    
    # Prepare for QLoRA if using quantization
    if QUANT_BITS in (4, 8):
        base_model = prepare_model_for_kbit_training(base_model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Compile model if requested and supported
    if os.getenv("TORCH_COMPILE", "false").lower() == "true":
        if torch.__version__ >= "2.0.0" and DEVICE == "cuda":
            model = torch.compile(model)
            log_event("model_compiled")
        else:
            log_event("model_compile_skipped", 
                      reason="Requires PyTorch 2.0+ and CUDA device")
    
    log_event("model_loading_completed", 
              parameters=model.num_parameters(),
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    return model

def ensure_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Ensure model and tokenizer are loaded.
    
    Returns:
        Tuple of (tokenizer, model)
    """
    global _model, _tokenizer
    
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = load_model()
        _model.eval()
        
        # Handle special token settings
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
            
    return _tokenizer, _model

def reload_adapter() -> None:
    """
    Reload the LoRA adapter if it exists.
    """
    if _model is not None and os.path.exists(ADAPTER_PATH):
        try:
            _model.load_adapter(ADAPTER_PATH, adapter_name="default", is_trainable=False)
            log_event("adapter_reloaded", path=ADAPTER_PATH)
        except Exception as e:
            log_event("adapter_reload_failed", error=str(e))

def generate_response(
    prompt: str, 
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Generate a response from the model.
    
    Args:
        prompt: Input text
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling parameter
        
    Returns:
        Generated text response
    """
    tokenizer, model = ensure_model()
    reload_adapter()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Extract only the new tokens (not the prompt)
    response = tokenizer.decode(
        outputs[0][input_length:], 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    return response

def train_lora(
    examples: list,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    epochs: int = 6
) -> None:
    """
    Train the LoRA adapter on a batch of examples with proper tensor handling.

    Args:
        examples: List of (prompt, response, source) tuples
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        epochs: Number of training epochs
    """
    import torch
    import os
    import time
    import shutil
    from pathlib import Path

    tokenizer, model = ensure_model()
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    log_event("training_started",
              examples=len(examples),
              epochs=epochs,
              learning_rate=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0

        # Process examples in batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            optimizer.zero_grad()

            batch_loss = 0
            for prompt, response, source in batch:
                try:
                    # Create a single training sequence: prompt + response
                    # This avoids batch size mismatches between input and target
                    combined_text = f"{prompt} {tokenizer.eos_token} {response}"

                    # Tokenize the combined text
                    encodings = tokenizer(
                        combined_text,
                        truncation=True,
                        max_length=2048,  # Adjust based on your model's context window
                        return_tensors="pt"
                    ).to(model.device)

                    # Create input_ids and labels tensors from the same sequence
                    input_ids = encodings.input_ids

                    # Copy input_ids to labels
                    labels = input_ids.clone()

                    # Mask the prompt part in labels (set to -100 to ignore in loss calculation)
                    # First, find where the prompt ends
                    prompt_tokens = tokenizer(
                        prompt,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.size(1)

                    # Set prompt tokens to -100 in labels to ignore them in loss calculation
                    labels[0, :prompt_tokens] = -100

                    # Calculate loss - only on the response part
                    outputs = model(
                        input_ids=input_ids,
                        labels=labels
                    )

                    loss = outputs.loss
                    batch_loss += loss.item()

                    # Backward pass
                    loss.backward()

                except Exception as e:
                    log_event("training_example_error",
                             prompt_length=len(prompt),
                             response_length=len(response),
                             error=str(e))
                    continue

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
            total_loss += batch_loss

        log_event("epoch_completed",
                  epoch=epoch+1,
                  loss=total_loss/len(examples))

    # Create a unique temporary path
    os.makedirs(os.path.dirname(ADAPTER_PATH), exist_ok=True)
    unique_temp_path = f"{ADAPTER_PATH}.tmp.{int(time.time())}"

    # Save the adapter to the temp path
    log_event("saving_adapter", temp_path=unique_temp_path)
    model.save_pretrained(unique_temp_path, safe_serialization=True)

    # Check if the old adapter exists and handle it appropriately
    if os.path.exists(ADAPTER_PATH):
        log_event("removing_old_adapter", path=ADAPTER_PATH)
        if os.path.isdir(ADAPTER_PATH):
            shutil.rmtree(ADAPTER_PATH)
        else:
            os.remove(ADAPTER_PATH)

    # Move the new adapter into place
    log_event("moving_adapter", from_path=unique_temp_path, to_path=ADAPTER_PATH)
    shutil.move(unique_temp_path, ADAPTER_PATH)

    model.eval()
    log_event("training_completed", adapter_path=ADAPTER_PATH)
