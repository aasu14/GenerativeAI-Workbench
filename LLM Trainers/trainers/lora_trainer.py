"""
Low-Rank Adaptation (LoRA) Trainer
For efficient fine-tuning with LoRA
"""

import logging
from typing import Dict, Any, Optional
from datasets import Dataset

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class LoRATrainer(BaseTrainer):
    """Low-Rank Adaptation Trainer for LLMs"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = kwargs.get("max_seq_length", 512)
        self.lora_r = kwargs.get("lora_r", 16)
        self.lora_alpha = kwargs.get("lora_alpha", 32)
        self.lora_dropout = kwargs.get("lora_dropout", 0.1)
        self.target_modules = kwargs.get("target_modules", None)
        self.bias = kwargs.get("bias", "none")
        self.use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", True)
        
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset for LoRA training"""
        logger.info("Preparing dataset for LoRA training")
        
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
            desc="Formatting dataset for LoRA"
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
    
    def create_trainer(self) -> Trainer:
        """Create LoRA trainer"""
        logger.info("Creating LoRA trainer")
        
        # Apply LoRA configuration
        self.model = self._apply_lora()
        
        # Enable gradient checkpointing if requested
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            **self.config
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("LoRA trainer created successfully")
        return trainer
    
    def _apply_lora(self):
        """Apply LoRA configuration to the model"""
        logger.info("Applying LoRA configuration")
        
        # Set default target modules if not specified
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "w1", "w2", "w3", "c_attn", "c_proj", "c_fc"
            ]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
        )
        
        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get LoRA-specific default configuration"""
        config = super()._get_default_config()
        config.update({
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
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
            "optim": "adamw_torch",
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
            "target_modules": self.target_modules,
            "bias": self.bias,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }
    
    def update_lora_config(self, **kwargs):
        """Update LoRA configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"LoRA configuration updated: {key} = {value}")
            else:
                logger.warning(f"Unknown LoRA configuration parameter: {key}")
    
    def set_target_modules(self, target_modules: list):
        """Set target modules for LoRA"""
        self.target_modules = target_modules
        logger.info(f"Target modules set to: {target_modules}")
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        if hasattr(self.model, 'print_trainable_parameters'):
            return self.model.num_trainable_parameters
        return 0
    
    def save_lora_weights(self, output_dir: str = None):
        """Save only the LoRA weights"""
        if output_dir is None:
            output_dir = self.output_dir / "lora_weights"
        
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving LoRA weights to {output_dir}")
        self.model.save_pretrained(str(output_dir))
        logger.info("LoRA weights saved successfully")
