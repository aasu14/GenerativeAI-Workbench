#!/usr/bin/env python3
"""
Comprehensive Demo Script for LLM Trainers
Demonstrates all trainer types with sample data
"""

import json
import tempfile
import os
from pathlib import Path
from llm_trainers import TrainerFactory

def create_sample_datasets():
    """Create sample datasets for different training approaches"""
    
    # Create temp directory for datasets
    temp_dir = tempfile.mkdtemp()
    
    # 1. SFT Dataset (Instruction-following)
    sft_data = [
        {
            "instruction": "Explain what machine learning is in simple terms.",
            "output": "Machine learning is a type of artificial intelligence that allows computers to learn and improve from experience without being explicitly programmed. Think of it like teaching a computer to recognize patterns, similar to how humans learn from examples."
        },
        {
            "instruction": "Write a short poem about coding.",
            "output": "Lines of logic flow like streams,\nFunctions dance in digital dreams,\nBugs may hide but we'll prevail,\nIn this coding fairy tale."
        },
        {
            "instruction": "What are the benefits of renewable energy?",
            "output": "Renewable energy offers numerous benefits including reduced greenhouse gas emissions, lower air pollution, energy security through domestic resources, job creation in clean energy sectors, and long-term cost savings despite higher initial investments."
        }
    ]
    
    sft_path = os.path.join(temp_dir, "sft_dataset.json")
    with open(sft_path, 'w') as f:
        json.dump(sft_data, f, indent=2)
    
    # 2. DPO Dataset (Preference learning)
    dpo_data = [
        {
            "prompt": "Explain quantum computing",
            "chosen": "Quantum computing is a revolutionary technology that uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously, enabling them to solve complex problems much faster than traditional computers.",
            "rejected": "Quantum computing is just a faster version of regular computers. It works the same way but with better hardware."
        },
        {
            "prompt": "Describe the benefits of exercise",
            "chosen": "Exercise provides numerous physical and mental health benefits including improved cardiovascular health, stronger muscles and bones, better mood through endorphin release, reduced stress and anxiety, improved sleep quality, enhanced cognitive function, and increased energy levels throughout the day.",
            "rejected": "Exercise is good for you because it makes you tired and sweaty. It's basically just moving around a lot."
        }
    ]
    
    dpo_path = os.path.join(temp_dir, "dpo_dataset.json")
    with open(dpo_path, 'w') as f:
        json.dump(dpo_data, f, indent=2)
    
    # 3. RLHF Dataset (Prompt-based)
    rlhf_data = [
        {"query": "How do I learn Python programming?"},
        {"query": "What are the best practices for machine learning?"},
        {"query": "Explain the concept of neural networks"},
        {"query": "How can I improve my productivity at work?"}
    ]
    
    rlhf_path = os.path.join(temp_dir, "rlhf_dataset.json")
    with open(rlhf_path, 'w') as f:
        json.dump(rlhf_data, f, indent=2)
    
    return {
        "sft": sft_path,
        "dpo": dpo_path,
        "rlhf": rlhf_path,
        "temp_dir": temp_dir
    }

def demo_sft_training():
    """Demonstrate SFT training"""
    print("\n" + "="*60)
    print("DEMO: Supervised Fine-Tuning (SFT) Training")
    print("="*60)
    
    try:
        # Create sample dataset
        datasets = create_sample_datasets()
        
        print("Creating SFT trainer...")
        sft_trainer = TrainerFactory.create_trainer(
            trainer_type="sft",
            model_name="microsoft/DialoGPT-medium",
            dataset_path=datasets["sft"],
            output_dir="./demo_output/sft",
            max_seq_length=256,
            learning_rate=5e-5,
            num_train_epochs=1,  # Demo with 1 epoch
            per_device_train_batch_size=1,
            logging_steps=1,
            save_steps=10
        )
        
        print("✓ SFT trainer created successfully!")
        print(f"Configuration: {sft_trainer.get_config()}")
        
        # Show dataset info
        print(f"Dataset loaded: {len(sft_trainer.dataset)} examples")
        
        # Uncomment to actually run training (requires GPU and model download)
        # print("Starting SFT training...")
        # sft_trainer.train()
        
        return sft_trainer
        
    except Exception as e:
        print(f"✗ Error in SFT demo: {e}")
        return None

def demo_dpo_training():
    """Demonstrate DPO training"""
    print("\n" + "="*60)
    print("DEMO: Direct Preference Optimization (DPO) Training")
    print("="*60)
    
    try:
        # Create sample dataset
        datasets = create_sample_datasets()
        
        print("Creating DPO trainer...")
        dpo_trainer = TrainerFactory.create_trainer(
            trainer_type="dpo",
            model_name="microsoft/DialoGPT-medium",
            dataset_path=datasets["dpo"],
            output_dir="./demo_output/dpo",
            beta=0.1,
            max_prompt_length=256,
            max_length=512,
            learning_rate=1e-5,
            num_train_epochs=1,  # Demo with 1 epoch
            per_device_train_batch_size=1,
            use_peft=True
        )
        
        print("✓ DPO trainer created successfully!")
        print(f"Configuration: {dpo_trainer.get_config()}")
        
        # Show dataset info
        print(f"Dataset loaded: {len(dpo_trainer.dataset)} preference examples")
        
        # Uncomment to actually run training (requires GPU and model download)
        # print("Starting DPO training...")
        # dpo_trainer.train()
        
        return dpo_trainer
        
    except Exception as e:
        print(f"✗ Error in DPO demo: {e}")
        return None

def demo_qlora_training():
    """Demonstrate QLoRA training"""
    print("\n" + "="*60)
    print("DEMO: Quantized LoRA (QLoRA) Training")
    print("="*60)
    
    try:
        # Create sample dataset
        datasets = create_sample_datasets()
        
        print("Creating QLoRA trainer...")
        qlora_trainer = TrainerFactory.create_trainer(
            trainer_type="qlora",
            model_name="microsoft/DialoGPT-medium",
            dataset_path=datasets["sft"],  # Can use SFT dataset
            output_dir="./demo_output/qlora",
            lora_r=32,  # Smaller for demo
            lora_alpha=16,
            lora_dropout=0.1,
            use_4bit=True,
            use_nested_quant=True,
            max_seq_length=256,
            learning_rate=2e-4,
            num_train_epochs=1,  # Demo with 1 epoch
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True
        )
        
        print("✓ QLoRA trainer created successfully!")
        print(f"Configuration: {qlora_trainer.get_config()}")
        print(f"LoRA Configuration: {qlora_trainer.get_lora_config()}")
        
        # Show dataset info
        print(f"Dataset loaded: {len(qlora_trainer.dataset)} examples")
        
        # Uncomment to actually run training (requires GPU and model download)
        # print("Starting QLoRA training...")
        # qlora_trainer.train()
        
        return qlora_trainer
        
    except Exception as e:
        print(f"✗ Error in QLoRA demo: {e}")
        return None

def demo_rlhf_training():
    """Demonstrate RLHF training"""
    print("\n" + "="*60)
    print("DEMO: Reinforcement Learning from Human Feedback (RLHF)")
    print("="*60)
    
    try:
        # Create sample dataset
        datasets = create_sample_datasets()
        
        print("Creating RLHF trainer...")
        rlhf_trainer = TrainerFactory.create_trainer(
            trainer_type="rlhf",
            model_name="microsoft/DialoGPT-medium",
            dataset_path=datasets["rlhf"],
            output_dir="./demo_output/rlhf",
            max_prompt_length=256,
            max_length=512,
            use_peft=True,
            learning_rate=1e-5,
            num_train_epochs=1,  # Demo with 1 epoch
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4
        )
        
        print("✓ RLHF trainer created successfully!")
        print(f"Configuration: {rlhf_trainer.get_config()}")
        print(f"PPO Configuration: {rlhf_trainer.get_ppo_config()}")
        
        # Show dataset info
        print(f"Dataset loaded: {len(rlhf_trainer.dataset)} prompt examples")
        
        # Uncomment to actually run training (requires GPU and model download)
        # print("Starting RLHF training...")
        # rlhf_trainer.train()
        
        return rlhf_trainer
        
    except Exception as e:
        print(f"✗ Error in RLHF demo: {e}")
        return None

def demo_peft_training():
    """Demonstrate PEFT training with different methods"""
    print("\n" + "="*60)
    print("DEMO: Parameter-Efficient Fine-Tuning (PEFT)")
    print("="*60)
    
    try:
        # Create sample dataset
        datasets = create_sample_datasets()
        
        # Test different PEFT methods
        peft_methods = ["lora", "prefix", "prompt", "p_tuning"]
        
        for method in peft_methods:
            print(f"\n--- Testing {method.upper()} method ---")
            
            peft_trainer = TrainerFactory.create_trainer(
                trainer_type="peft",
                model_name="microsoft/DialoGPT-medium",
                dataset_path=datasets["sft"],
                output_dir=f"./demo_output/peft_{method}",
                peft_method=method,
                max_seq_length=256,
                learning_rate=1e-4,
                num_train_epochs=1,  # Demo with 1 epoch
                per_device_train_batch_size=1
            )
            
            print(f"✓ {method.upper()} trainer created successfully!")
            print(f"PEFT Configuration: {peft_trainer.get_peft_config()}")
            
            # Uncomment to actually run training (requires GPU and model download)
            # print(f"Starting {method.upper()} training...")
            # peft_trainer.train()
        
        return True
        
    except Exception as e:
        print(f"✗ Error in PEFT demo: {e}")
        return None

def show_trainer_comparison():
    """Show comparison of different trainer types"""
    print("\n" + "="*60)
    print("TRAINER COMPARISON")
    print("="*60)
    
    print("\nSupported Trainer Types:")
    supported_trainers = TrainerFactory.get_supported_trainers()
    
    for trainer_type in supported_trainers:
        try:
            info = TrainerFactory.get_trainer_info(trainer_type)
            print(f"\n• {info['name']} ({trainer_type})")
            print(f"  Description: {info['description']}")
            
            # Show key configuration differences
            config = info['default_config']
            if config:
                key_params = ['learning_rate', 'per_device_train_batch_size', 'gradient_accumulation_steps']
                print("  Key Parameters:")
                for param in key_params:
                    if param in config:
                        print(f"    {param}: {config[param]}")
                        
        except Exception as e:
            print(f"• {trainer_type.upper()}: Error getting info - {e}")

def main():
    """Main demo function"""
    print("🚀 LLM Trainers Comprehensive Demo")
    print("="*60)
    print("This demo showcases all available trainer types")
    print("Note: Training is not executed by default to avoid model downloads")
    print("Uncomment the training lines in each demo function to run actual training")
    
    # Create output directory
    os.makedirs("./demo_output", exist_ok=True)
    
    # Run demos
    trainers = {}
    
    # SFT Demo
    trainers['sft'] = demo_sft_training()
    
    # DPO Demo
    trainers['dpo'] = demo_dpo_training()
    
    # QLoRA Demo
    trainers['qlora'] = demo_qlora_training()
    
    # RLHF Demo
    trainers['rlhf'] = demo_rlhf_training()
    
    # PEFT Demo
    demo_peft_training()
    
    # Show comparison
    show_trainer_comparison()
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    successful_demos = sum(1 for t in trainers.values() if t is not None)
    total_demos = len(trainers)
    
    print(f"Successful demos: {successful_demos}/{total_demos}")
    
    if successful_demos == total_demos:
        print("🎉 All demos completed successfully!")
        print("\nTo run actual training:")
        print("1. Ensure you have a GPU available")
        print("2. Uncomment the training lines in each demo function")
        print("3. Run the demo again")
    else:
        print("⚠️  Some demos failed. Check the error messages above.")
    
    print(f"\nOutput directories created in: ./demo_output/")
    print("Check the individual trainer directories for results and logs.")

if __name__ == "__main__":
    main()

