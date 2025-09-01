"""
Reinforcement Learning from Human Feedback (RLHF) Trainer
For reward modeling and policy optimization
"""

import logging
from typing import Dict, Any, Optional
from datasets import Dataset

import torch
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model
)
from peft import LoraConfig, get_peft_model

from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class RLHFTrainer(BaseTrainer):
    """Reinforcement Learning from Human Feedback Trainer for LLMs"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_model_name = kwargs.get("reward_model_name", None)
        self.ppo_config = kwargs.get("ppo_config", {})
        self.use_peft = kwargs.get("use_peft", True)
        self.max_prompt_length = kwargs.get("max_prompt_length", 512)
        self.max_length = kwargs.get("max_length", 1024)
        
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset for RLHF training"""
        logger.info("Preparing dataset for RLHF training")
        
        def format_prompt(example):
            """Format prompt data for RLHF training"""
            if "prompt" in example:
                return {"query": example["prompt"]}
            elif "instruction" in example:
                return {"query": example["instruction"]}
            elif "query" in example:
                return {"query": example["query"]}
            elif "text" in example:
                return {"query": example["text"]}
            else:
                # Try to find any text field
                text_fields = [k for k, v in example.items() if isinstance(v, str) and len(v) > 10]
                if text_fields:
                    return {"query": example[text_fields[0]]}
                else:
                    return {"query": str(example)}
        
        # Apply formatting
        formatted_dataset = self.dataset.map(
            format_prompt,
            remove_columns=self.dataset.column_names,
            desc="Formatting dataset for RLHF"
        )
        
        # Filter out invalid examples
        def is_valid(example):
            return len(example["query"]) > 0 and len(example["query"]) < 1000
        
        filtered_dataset = formatted_dataset.filter(
            is_valid,
            desc="Filtering valid prompt examples"
        )
        
        logger.info(f"Dataset prepared: {len(filtered_dataset)} prompt examples")
        return filtered_dataset
    
    def create_trainer(self) -> PPOTrainer:
        """Create RLHF trainer using PPO"""
        logger.info("Creating RLHF trainer with PPO")
        
        # Apply PEFT if enabled
        if self.use_peft:
            self.model = self._apply_peft()
        
        # Create value head model for PPO
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.config.get("use_fp16", True) else torch.float32,
            device_map="auto" if self.config.get("use_device_map", True) else None,
        )
        
        # Create reference model
        ref_model = create_reference_model(model)
        
        # PPO configuration
        ppo_config = PPOConfig(
            learning_rate=self.config.get("learning_rate", 1e-5),
            batch_size=self.config.get("per_device_train_batch_size", 1),
            mini_batch_size=1,
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            ppo_epochs=4,
            seed=42,
            init_kl_coef=0.2,
            adap_kl_ctrl=True,
            **self.ppo_config
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            **{k: v for k, v in self.config.items() if k not in ppo_config.__dict__}
        )
        
        # Create PPO trainer
        trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            args=training_args,
            max_prompt_length=self.max_prompt_length,
            max_length=self.max_length,
        )
        
        logger.info("RLHF trainer created successfully")
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
        """Get RLHF-specific default configuration"""
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
    
    def set_reward_model(self, reward_model_name: str):
        """Set the reward model for evaluation"""
        self.reward_model_name = reward_model_name
        logger.info(f"Reward model set to: {reward_model_name}")
    
    def get_ppo_config(self) -> Dict[str, Any]:
        """Get current PPO configuration"""
        return self.ppo_config.copy()
    
    def update_ppo_config(self, **kwargs):
        """Update PPO configuration parameters"""
        self.ppo_config.update(kwargs)
        logger.info(f"PPO configuration updated: {kwargs}")

