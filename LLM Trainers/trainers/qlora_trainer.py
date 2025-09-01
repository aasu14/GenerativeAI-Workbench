"""
Quantized Low-Rank Adaptation (QLoRA) Trainer
For efficient fine-tuning with quantization
"""

import logging
from typing import Dict, Any, Optional
from datasets import Dataset

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer as TRLSFTTrainer

from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class QLoRATrainer(BaseTrainer):
    """Quantized LoRA Trainer for efficient LLM fine-tuning"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = kwargs.get("max_seq_length", 512)
        self.packing = kwargs.get("packing", False)
        self.lora_r = kwargs.get("lora_r", 64)
        self.lora_alpha = kwargs.get("lora_alpha", 16)
        self.lora_dropout = kwargs.get("lora_dropout", 0.1)
        self.use_4bit = kwargs.get("use_4bit", True)
        self.use_nested_quant = kwargs.get("use_nested_quant", True)
        self.bnb_4bit_compute_dtype = kwargs.get("bnb_4bit_compute_dtype", torch.float16)
        
    def _load_model_and_tokenizer(self):
        """Load the base model and tokenizer with quantization"""
        logger.info(f"Loading model and tokenizer from {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configure quantization
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=self.use_nested_quant,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                
                # Prepare model for k-bit training
                self.model = prepare_model_for_kbit_training(self.model)
                
            else:
                # Load without quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model and tokenizer loaded successfully with QLoRA configuration")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset for QLoRA training"""
        logger.info("Preparing dataset for QLoRA training")
        
        def format_instruction(example):
            """Format instruction data for training"""
            if "instruction" in example and "output" in example:
                # Instruction-following format
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            elif "prompt" in example and "completion" in example:
                # Prompt-completion format
                text = f"{example['prompt']}{example['completion']}"
            elif "text" in example:
                # Plain text format
                text = example["text"]
            else:
                # Try to find any text field
                text_fields = [k for k, v in example.items() if isinstance(v, str) and len(v) > 10]
                if text_fields:
                    text = example[text_fields[0]]
                else:
                    text = str(example)
            
            return {"text": text}
        
        # Apply formatting
        formatted_dataset = self.dataset.map(
            format_instruction,
            remove_columns=self.dataset.column_names,
            desc="Formatting dataset for QLoRA"
        )
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_seq_length,
                return_tensors=None,
            )
        
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            remove_columns=formatted_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        logger.info(f"Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def create_trainer(self) -> TRLSFTTrainer:
        """Create QLoRA trainer using TRL SFT"""
        logger.info("Creating QLoRA trainer")
        
        # Apply LoRA configuration
        self.model = self._apply_lora()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            **self.config
        )
        
        # Create SFT trainer with QLoRA
        trainer = TRLSFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=self.max_seq_length,
            packing=self.packing,
            dataset_text_field="text",
        )
        
        logger.info("QLoRA trainer created successfully")
        return trainer
    
    def _apply_lora(self):
        """Apply LoRA configuration to the model"""
        logger.info("Applying LoRA configuration")
        
        # Define target modules based on model architecture
        target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "w1", "w2", "w3", "c_attn", "c_proj", "c_fc"
        ]
        
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get QLoRA-specific default configuration"""
        config = super()._get_default_config()
        config.update({
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "warmup_ratio": 0.1,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 100,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": True,
            "dataloader_pin_memory": False,
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 0.3,
        })
        return config
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get current LoRA configuration"""
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_4bit": self.use_4bit,
            "use_nested_quant": self.use_nested_quant,
        }
    
    def update_lora_config(self, **kwargs):
        """Update LoRA configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"LoRA configuration updated: {key} = {value}")
            else:
                logger.warning(f"Unknown LoRA configuration parameter: {key}")
