"""
Base Trainer Class
All LLM trainers inherit from this base class
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import wandb

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base class for all LLM trainers"""
    
    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str = "./output",
        **kwargs
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        # Configuration
        self.config = self._get_default_config()
        self.config.update(kwargs)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._load_model_and_tokenizer()
        self._load_dataset()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "eval_steps": 100,
            "save_steps": 500,
            "save_total_limit": 2,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "push_to_hub": False,
            "report_to": "wandb",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
    
    def _load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading model and tokenizer from {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.config.get("use_fp16", True) else torch.float32,
                device_map="auto" if self.config.get("use_device_map", True) else None,
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_dataset(self):
        """Load and prepare the dataset"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        try:
            if self.dataset_path.endswith('.json'):
                self.dataset = load_dataset('json', data_files=self.dataset_path)
            elif self.dataset_path.endswith('.csv'):
                self.dataset = load_dataset('csv', data_files=self.dataset_path)
            else:
                self.dataset = load_dataset(self.dataset_path)
                
            logger.info(f"Dataset loaded: {self.dataset}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    @abstractmethod
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset for training - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def create_trainer(self) -> Trainer:
        """Create the specific trainer - must be implemented by subclasses"""
        pass
    
    def train(self):
        """Execute the training process"""
        logger.info("Starting training process")
        
        try:
            # Prepare dataset
            prepared_dataset = self.prepare_dataset()
            
            # Create trainer
            self.trainer = self.create_trainer()
            
            # Start training
            logger.info("Starting training...")
            self.trainer.train()
            
            # Save final model
            self.save_model()
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self):
        """Save the trained model"""
        save_path = self.output_dir / "final_model"
        logger.info(f"Saving model to {save_path}")
        
        try:
            self.trainer.save_model(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
            
            # Save training config
            with open(save_path / "training_config.json", "w") as f:
                json.dump(self.config, f, indent=2)
                
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def evaluate(self):
        """Evaluate the trained model"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Run train() first.")
        
        logger.info("Starting evaluation...")
        results = self.trainer.evaluate()
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        self.config.update(kwargs)
        logger.info(f"Configuration updated: {kwargs}")

