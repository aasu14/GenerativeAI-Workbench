"""
Direct Preference Optimization (DPO) Trainer
For preference learning and alignment
"""

import logging
from typing import Dict, Any, Optional
from datasets import Dataset

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM
)
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model

from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization Trainer for LLMs"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = kwargs.get("beta", 0.1)
        self.max_prompt_length = kwargs.get("max_prompt_length", 512)
        self.max_length = kwargs.get("max_length", 1024)
        self.use_peft = kwargs.get("use_peft", True)
        
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset for DPO training"""
        logger.info("Preparing dataset for DPO training")
        
        def format_preference(example):
            """Format preference data for DPO training"""
            if "prompt" in example and "chosen" in example and "rejected" in example:
                # Standard preference format
                return {
                    "prompt": example["prompt"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"]
                }
            elif "instruction" in example and "chosen" in example and "rejected" in example:
                # Instruction-based preference format
                return {
                    "prompt": example["instruction"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"]
                }
            elif "query" in example and "positive" in example and "negative" in example:
                # Query-based preference format
                return {
                    "prompt": example["query"],
                    "chosen": example["positive"],
                    "rejected": example["negative"]
                }
            else:
                # Try to infer preference format
                keys = list(example.keys())
                if len(keys) >= 3:
                    # Assume first key is prompt, second is chosen, third is rejected
                    return {
                        "prompt": example[keys[0]],
                        "chosen": example[keys[1]],
                        "rejected": example[keys[2]]
                    }
                else:
                    raise ValueError(f"Cannot infer preference format from example: {example}")
        
        # Apply formatting
        formatted_dataset = self.dataset.map(
            format_preference,
            remove_columns=self.dataset.column_names,
            desc="Formatting dataset for DPO"
        )
        
        # Filter out invalid examples
        def is_valid(example):
            return (
                len(example["prompt"]) > 0 and
                len(example["chosen"]) > 0 and
                len(example["rejected"]) > 0 and
                example["chosen"] != example["rejected"]
            )
        
        filtered_dataset = formatted_dataset.filter(
            is_valid,
            desc="Filtering valid preference examples"
        )
        
        logger.info(f"Dataset prepared: {len(filtered_dataset)} preference examples")
        return filtered_dataset
    
    def create_trainer(self) -> DPOTrainer:
        """Create DPO trainer"""
        logger.info("Creating DPO trainer")
        
        # Apply PEFT if enabled
        if self.use_peft:
            self.model = self._apply_peft()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            **self.config
        )
        
        # Create DPO trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Use the same model as reference
            args=training_args,
            beta=self.beta,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            max_prompt_length=self.max_prompt_length,
            max_length=self.max_length,
            peft_config=None,  # Already applied if use_peft=True
        )
        
        logger.info("DPO trainer created successfully")
        return trainer
    
    def _apply_peft(self):
        """Apply PEFT configuration to the model"""
        logger.info("Applying PEFT configuration")
        
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get DPO-specific default configuration"""
        config = super()._get_default_config()
        config.update({
            "learning_rate": 1e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
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
            "remove_unused_columns": False,
        })
        return config

