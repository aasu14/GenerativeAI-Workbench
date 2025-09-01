"""
Configuration Example
Demonstrates how to use configuration files with the trainer factory
"""

import json
import os
from pathlib import Path
from llm_trainers import TrainerFactory

def create_sample_configs():
    """Create sample configuration files for different trainer types"""
    
    # Create configs directory
    configs_dir = Path("./configs")
    configs_dir.mkdir(exist_ok=True)
    
    # SFT Configuration
    sft_config = {
        "trainer_type": "sft",
        "model_name": "microsoft/DialoGPT-medium",
        "dataset_path": "path/to/your/dataset.json",
        "output_dir": "./output/sft",
        "max_seq_length": 512,
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 100,
        "fp16": True
    }
    
    with open(configs_dir / "sft_config.json", "w") as f:
        json.dump(sft_config, f, indent=2)
    
    # DPO Configuration
    dpo_config = {
        "trainer_type": "dpo",
        "model_name": "microsoft/DialoGPT-medium",
        "dataset_path": "path/to/your/preference_dataset.json",
        "output_dir": "./output/dpo",
        "beta": 0.1,
        "max_prompt_length": 512,
        "max_length": 1024,
        "learning_rate": 1e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "use_peft": True,
        "fp16": True
    }
    
    with open(configs_dir / "dpo_config.json", "w") as f:
        json.dump(dpo_config, f, indent=2)
    
    # QLoRA Configuration
    qlora_config = {
        "trainer_type": "qlora",
        "model_name": "microsoft/DialoGPT-medium",
        "dataset_path": "path/to/your/dataset.json",
        "output_dir": "./output/qlora",
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "use_4bit": True,
        "use_nested_quant": True,
        "max_seq_length": 512,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "fp16": True
    }
    
    with open(configs_dir / "qlora_config.json", "w") as f:
        json.dump(qlora_config, f, indent=2)
    
    # RLHF Configuration
    rlhf_config = {
        "trainer_type": "rlhf",
        "model_name": "microsoft/DialoGPT-medium",
        "dataset_path": "path/to/your/prompt_dataset.json",
        "output_dir": "./output/rlhf",
        "max_prompt_length": 512,
        "max_length": 1024,
        "use_peft": True,
        "ppo_config": {
            "target_kl": 0.1,
            "ppo_epochs": 4,
            "init_kl_coef": 0.2
        },
        "learning_rate": 1e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "fp16": True
    }
    
    with open(configs_dir / "rlhf_config.json", "w") as f:
        json.dump(rlhf_config, f, indent=2)
    
    print("Sample configuration files created in ./configs/")

def load_and_train_from_config():
    """Load trainer from configuration file and start training"""
    
    config_file = "./configs/sft_config.json"
    
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        print("Run create_sample_configs() first to create sample configs")
        return
    
    try:
        print(f"Loading trainer from configuration: {config_file}")
        
        # Create trainer from config file
        trainer = TrainerFactory.create_trainer_from_config(config_file)
        
        print("Trainer created successfully from configuration!")
        print(f"Trainer type: {trainer.__class__.__name__}")
        print(f"Configuration: {trainer.get_config()}")
        
        # Uncomment to start training
        # trainer.train()
        
    except Exception as e:
        print(f"Error creating trainer from config: {e}")

def main():
    """Main function"""
    print("=== Configuration Example ===")
    
    # Create sample configuration files
    print("Creating sample configuration files...")
    create_sample_configs()
    
    print("\n" + "="*50 + "\n")
    
    # Load and train from configuration
    print("Loading trainer from configuration file...")
    load_and_train_from_config()
    
    print("\n" + "="*50 + "\n")
    
    # Show supported trainer types
    print("Supported trainer types:")
    supported_trainers = TrainerFactory.get_supported_trainers()
    for trainer_type in supported_trainers:
        info = TrainerFactory.get_trainer_info(trainer_type)
        print(f"- {trainer_type.upper()}: {info['description']}")

if __name__ == "__main__":
    main()

