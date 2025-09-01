"""
Adaptive Low-Rank Adaptation (AdaLoRA) Trainer
For adaptive rank allocation during fine-tuning
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
from peft import AdaLoraConfig, get_peft_model, TaskType

from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class AdaLoRATrainer(BaseTrainer):
    """Adaptive LoRA Trainer for LLMs with dynamic rank allocation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = kwargs.get("max_seq_length", 512)
        self.target_r = kwargs.get("target_r", 16)
        self.init_r = kwargs.get("init_r", 8)
        self.beta1 = kwargs.get("beta1", 0.85)
        self.beta2 = kwargs.get("beta2", 0.85)
        self.target_modules = kwargs.get("target_modules", None)
        self.bias = kwargs.get("bias", "none")
        self.use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", True)
        self.rank_pattern = kwargs.get("rank_pattern", None)
        
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset for AdaLoRA training"""
        logger.info("Preparing dataset for AdaLoRA training")
        
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
            desc="Formatting dataset for AdaLoRA"
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
        """Create AdaLoRA trainer"""
        logger.info("Creating AdaLoRA trainer")
        
        # Apply AdaLoRA configuration
        self.model = self._apply_adalora()
        
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
        
        logger.info("AdaLoRA trainer created successfully")
        return trainer
    
    def _apply_adalora(self):
        """Apply AdaLoRA configuration to the model"""
        logger.info("Applying AdaLoRA configuration")
        
        # Set default target modules if not specified
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "w1", "w2", "w3", "c_attn", "c_proj", "c_fc"
            ]
        
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_r=self.target_r,
            init_r=self.init_r,
            beta1=self.beta1,
            beta2=self.beta2,
            target_modules=self.target_modules,
            bias=self.bias,
            rank_pattern=self.rank_pattern,
        )
        
        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get AdaLoRA-specific default configuration"""
        config = super()._get_default_config()
        config.update({
            "learning_rate": 1e-4,
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
    
    def get_adalora_config(self) -> Dict[str, Any]:
        """Get current AdaLoRA configuration"""
        return {
            "target_r": self.target_r,
            "init_r": self.init_r,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "rank_pattern": self.rank_pattern,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }
    
    def update_adalora_config(self, **kwargs):
        """Update AdaLoRA configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"AdaLoRA configuration updated: {key} = {value}")
            else:
                logger.warning(f"Unknown AdaLoRA configuration parameter: {key}")
    
    def set_target_modules(self, target_modules: list):
        """Set target modules for AdaLoRA"""
        self.target_modules = target_modules
        logger.info(f"Target modules set to: {target_modules}")
    
    def set_rank_pattern(self, rank_pattern: dict):
        """Set rank pattern for AdaLoRA"""
        self.rank_pattern = rank_pattern
        logger.info(f"Rank pattern set to: {rank_pattern}")
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        if hasattr(self.model, 'print_trainable_parameters'):
            return self.model.num_trainable_parameters
        return 0
    
    def get_rank_allocation(self) -> dict:
        """Get current rank allocation across modules"""
        if hasattr(self.model, 'get_rank_allocation'):
            return self.model.get_rank_allocation()
        return {}
    
    def save_adalora_weights(self, output_dir: str = None):
        """Save only the AdaLoRA weights"""
        if output_dir is None:
            output_dir = self.output_dir / "adalora_weights"
        
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving AdaLoRA weights to {output_dir}")
        self.model.save_pretrained(str(output_dir))
        logger.info("AdaLoRA weights saved successfully")
    
    def update_rank_pattern(self, new_pattern: dict):
        """Update rank pattern during training"""
        if hasattr(self.model, 'update_rank_pattern'):
            self.model.update_rank_pattern(new_pattern)
            logger.info(f"Rank pattern updated to: {new_pattern}")
        else:
            logger.warning("Model does not support dynamic rank pattern updates")
