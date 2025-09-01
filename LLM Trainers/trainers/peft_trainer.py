"""
Parameter-Efficient Fine-Tuning (PEFT) Trainer
For various PEFT methods including LoRA, Prefix Tuning, etc.
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
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    P_TuningConfig,
    get_peft_model,
    TaskType
)

from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class PEFTTrainer(BaseTrainer):
    """Parameter-Efficient Fine-Tuning Trainer for LLMs"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peft_method = kwargs.get("peft_method", "lora")
        self.max_seq_length = kwargs.get("max_seq_length", 512)
        
        # LoRA specific parameters
        self.lora_r = kwargs.get("lora_r", 16)
        self.lora_alpha = kwargs.get("lora_alpha", 32)
        self.lora_dropout = kwargs.get("lora_dropout", 0.1)
        
        # Prefix Tuning parameters
        self.prefix_length = kwargs.get("prefix_length", 20)
        
        # Prompt Tuning parameters
        self.prompt_length = kwargs.get("prompt_length", 20)
        
        # P-Tuning parameters
        self.prompt_encoder_hidden_size = kwargs.get("prompt_encoder_hidden_size", 128)
        
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset for PEFT training"""
        logger.info("Preparing dataset for PEFT training")
        
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
            desc="Formatting dataset for PEFT"
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
        """Create PEFT trainer"""
        logger.info(f"Creating PEFT trainer with method: {self.peft_method}")
        
        # Apply PEFT configuration
        self.model = self._apply_peft()
        
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
        
        logger.info("PEFT trainer created successfully")
        return trainer
    
    def _apply_peft(self):
        """Apply PEFT configuration to the model"""
        logger.info(f"Applying {self.peft_method.upper()} configuration")
        
        if self.peft_method.lower() == "lora":
            peft_config = self._get_lora_config()
        elif self.peft_method.lower() == "prefix":
            peft_config = self._get_prefix_config()
        elif self.peft_method.lower() == "prompt":
            peft_config = self._get_prompt_config()
        elif self.peft_method.lower() == "p_tuning":
            peft_config = self._get_p_tuning_config()
        else:
            raise ValueError(f"Unsupported PEFT method: {self.peft_method}")
        
        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def _get_lora_config(self):
        """Get LoRA configuration"""
        target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "w1", "w2", "w3", "c_attn", "c_proj", "c_fc"
        ]
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
    
    def _get_prefix_config(self):
        """Get Prefix Tuning configuration"""
        return PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.prefix_length,
            encoder_hidden_size=self.prompt_encoder_hidden_size,
        )
    
    def _get_prompt_config(self):
        """Get Prompt Tuning configuration"""
        return PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.prompt_length,
            encoder_hidden_size=self.prompt_encoder_hidden_size,
        )
    
    def _get_p_tuning_config(self):
        """Get P-Tuning configuration"""
        return P_TuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.prompt_length,
            encoder_hidden_size=self.prompt_encoder_hidden_size,
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get PEFT-specific default configuration"""
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
        })
        return config
    
    def get_peft_config(self) -> Dict[str, Any]:
        """Get current PEFT configuration"""
        return {
            "peft_method": self.peft_method,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "prefix_length": self.prefix_length,
            "prompt_length": self.prompt_length,
            "prompt_encoder_hidden_size": self.prompt_encoder_hidden_size,
        }
    
    def update_peft_config(self, **kwargs):
        """Update PEFT configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"PEFT configuration updated: {key} = {value}")
            else:
                logger.warning(f"Unknown PEFT configuration parameter: {key}")
    
    def set_peft_method(self, method: str):
        """Set PEFT method"""
        supported_methods = ["lora", "prefix", "prompt", "p_tuning"]
        if method.lower() not in supported_methods:
            raise ValueError(f"Unsupported PEFT method: {method}. Supported: {supported_methods}")
        
        self.peft_method = method.lower()
        logger.info(f"PEFT method set to: {self.peft_method}")

