"""
Basic Usage Example
Demonstrates how to use the LLM Trainers repository
"""

import os
from llm_trainers import TrainerFactory

def main():
    """Main function demonstrating basic usage"""
    
    # Example 1: SFT Training
    print("=== SFT Training Example ===")
    try:
        sft_trainer = TrainerFactory.create_trainer(
            trainer_type="sft",
            model_name="microsoft/DialoGPT-medium",
            dataset_path="path/to/your/dataset.json",
            output_dir="./output/sft",
            max_seq_length=512,
            learning_rate=5e-5
        )
        
        print("SFT Trainer created successfully!")
        print(f"Configuration: {sft_trainer.get_config()}")
        
        # Uncomment to start training
        # sft_trainer.train()
        
    except Exception as e:
        print(f"Error creating SFT trainer: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: DPO Training
    print("=== DPO Training Example ===")
    try:
        dpo_trainer = TrainerFactory.create_trainer(
            trainer_type="dpo",
            model_name="microsoft/DialoGPT-medium",
            dataset_path="path/to/your/preference_dataset.json",
            output_dir="./output/dpo",
            beta=0.1,
            max_prompt_length=512,
            max_length=1024
        )
        
        print("DPO Trainer created successfully!")
        print(f"Configuration: {dpo_trainer.get_config()}")
        
        # Uncomment to start training
        # dpo_trainer.train()
        
    except Exception as e:
        print(f"Error creating DPO trainer: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: QLoRA Training
    print("=== QLoRA Training Example ===")
    try:
        qlora_trainer = TrainerFactory.create_trainer(
            trainer_type="qlora",
            model_name="microsoft/DialoGPT-medium",
            dataset_path="path/to/your/dataset.json",
            output_dir="./output/qlora",
            lora_r=64,
            lora_alpha=16,
            use_4bit=True,
            learning_rate=2e-4
        )
        
        print("QLoRA Trainer created successfully!")
        print(f"Configuration: {qlora_trainer.get_config()}")
        print(f"LoRA Configuration: {qlora_trainer.get_lora_config()}")
        
        # Uncomment to start training
        # qlora_trainer.train()
        
    except Exception as e:
        print(f"Error creating QLoRA trainer: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: RLHF Training
    print("=== RLHF Training Example ===")
    try:
        rlhf_trainer = TrainerFactory.create_trainer(
            trainer_type="rlhf",
            model_name="microsoft/DialoGPT-medium",
            dataset_path="path/to/your/prompt_dataset.json",
            output_dir="./output/rlhf",
            max_prompt_length=512,
            max_length=1024,
            use_peft=True
        )
        
        print("RLHF Trainer created successfully!")
        print(f"Configuration: {rlhf_trainer.get_config()}")
        print(f"PPO Configuration: {rlhf_trainer.get_ppo_config()}")
        
        # Uncomment to start training
        # rlhf_trainer.train()
        
    except Exception as e:
        print(f"Error creating RLHF trainer: {e}")

if __name__ == "__main__":
    main()

