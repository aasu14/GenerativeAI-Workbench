"""
Trainer Factory
Creates the appropriate trainer based on the trainer type
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .trainers.sft_trainer import SFTTrainer
from .trainers.dpo_trainer import DPOTrainer
from .trainers.rlhf_trainer import RLHFTrainer
from .trainers.qlora_trainer import QLoRATrainer
from .trainers.peft_trainer import PEFTTrainer
from .trainers.lora_trainer import LoRATrainer
from .trainers.adalora_trainer import AdaLoRATrainer

logger = logging.getLogger(__name__)


class TrainerFactory:
    """Factory class for creating different types of LLM trainers"""
    
    # Mapping of trainer types to trainer classes
    TRAINER_REGISTRY = {
        "sft": SFTTrainer,
        "dpo": DPOTrainer,
        "rlhf": RLHFTrainer,
        "qlora": QLoRATrainer,
        "peft": PEFTTrainer,
        "lora": LoRATrainer,
        "adalora": AdaLoRATrainer,
    }
    
    @classmethod
    def create_trainer(
        cls,
        trainer_type: str,
        model_name: str,
        dataset_path: str,
        output_dir: str = "./output",
        **kwargs
    ):
        """
        Create a trainer instance based on the trainer type
        
        Args:
            trainer_type: Type of trainer to create
            model_name: Name/path of the base model
            dataset_path: Path to the training dataset
            output_dir: Directory to save outputs
            **kwargs: Additional arguments for the trainer
            
        Returns:
            Trainer instance
            
        Raises:
            ValueError: If trainer_type is not supported
        """
        trainer_type = trainer_type.lower()
        
        if trainer_type not in cls.TRAINER_REGISTRY:
            supported_types = list(cls.TRAINER_REGISTRY.keys())
            raise ValueError(
                f"Unsupported trainer type: {trainer_type}. "
                f"Supported types: {supported_types}"
            )
        
        trainer_class = cls.TRAINER_REGISTRY[trainer_type]
        logger.info(f"Creating {trainer_type.upper()} trainer")
        
        try:
            trainer = trainer_class(
                model_name=model_name,
                dataset_path=dataset_path,
                output_dir=output_dir,
                **kwargs
            )
            return trainer
            
        except Exception as e:
            logger.error(f"Failed to create {trainer_type} trainer: {e}")
            raise
    
    @classmethod
    def get_supported_trainers(cls) -> list:
        """Get list of supported trainer types"""
        return list(cls.TRAINER_REGISTRY.keys())
    
    @classmethod
    def get_trainer_info(cls, trainer_type: str) -> Dict[str, Any]:
        """Get information about a specific trainer type"""
        trainer_type = trainer_type.lower()
        
        if trainer_type not in cls.TRAINER_REGISTRY:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")
        
        trainer_class = cls.TRAINER_REGISTRY[trainer_type]
        
        # Get trainer description from docstring
        description = trainer_class.__doc__ or "No description available"
        
        # Get default config
        trainer_instance = trainer_class.__new__(trainer_class)
        if hasattr(trainer_instance, '_get_default_config'):
            default_config = trainer_instance._get_default_config()
        else:
            default_config = {}
        
        return {
            "name": trainer_type.upper(),
            "class": trainer_class.__name__,
            "description": description.strip(),
            "default_config": default_config,
            "supported_methods": getattr(trainer_instance, 'get_supported_methods', lambda: [])()
        }
    
    @classmethod
    def validate_trainer_config(cls, trainer_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and return validated configuration for a trainer type"""
        trainer_type = trainer_type.lower()
        
        if trainer_type not in cls.TRAINER_REGISTRY:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")
        
        # Get default config for validation
        trainer_class = cls.TRAINER_REGISTRY[trainer_type]
        trainer_instance = trainer_class.__new__(trainer_class)
        
        if hasattr(trainer_instance, '_get_default_config'):
            default_config = trainer_instance._get_default_config()
            
            # Validate required fields
            validated_config = default_config.copy()
            validated_config.update(config)
            
            # Log any unknown parameters
            unknown_params = set(config.keys()) - set(default_config.keys())
            if unknown_params:
                logger.warning(f"Unknown parameters for {trainer_type}: {unknown_params}")
            
            return validated_config
        
        return config
    
    @classmethod
    def create_trainer_from_config(cls, config_file: str):
        """
        Create a trainer from a configuration file
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            Trainer instance
        """
        import json
        
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract trainer type and required parameters
        trainer_type = config.pop("trainer_type")
        model_name = config.pop("model_name")
        dataset_path = config.pop("dataset_path")
        output_dir = config.pop("output_dir", "./output")
        
        # Create trainer with remaining config
        return cls.create_trainer(
            trainer_type=trainer_type,
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            **config
        )

