"""
Supervised Fine-Tuning (SFT) Trainer
For instruction-following and supervised learning tasks
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
from trl import SFTTrainer as TRLSFTTrainer

from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning Trainer for LLMs"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = kwargs.get("max_seq_length", 512)
        self.packing = kwargs.get("packing", False)
        
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset for SFT training"""
        logger.info("Preparing dataset for SFT training")
        
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
            desc="Formatting dataset for SFT"
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
        """Create SFT trainer using TRL"""
        logger.info("Creating SFT trainer")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            **self.config
        )
        
        # Create SFT trainer
        trainer = TRLSFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=self.max_seq_length,
            packing=self.packing,
            dataset_text_field="text",
        )
        
        logger.info("SFT trainer created successfully")
        return trainer
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get SFT-specific default configuration"""
        config = super()._get_default_config()
        config.update({
            "learning_rate": 5e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
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
        })
        return config

