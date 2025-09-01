#!/usr/bin/env python3
"""
Command Line Interface for LLM Trainers
Provides a convenient way to run training from the command line
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any

from .trainer_factory import TrainerFactory


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Trainers - Fine-tune Large Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SFT Training
  python -m llm_trainers.cli train --type sft --model microsoft/DialoGPT-medium --dataset data.json --output ./output

  # DPO Training
  python -m llm_trainers.cli train --type dpo --model microsoft/DialoGPT-medium --dataset preferences.json --output ./output

  # QLoRA Training
  python -m llm_trainers.cli train --type qlora --model microsoft/DialoGPT-medium --dataset data.json --output ./output --lora-r 64

  # From Configuration File
  python -m llm_trainers.cli train --config config.json

  # List Supported Trainers
  python -m llm_trainers.cli list

  # Get Trainer Info
  python -m llm_trainers.cli info --type sft
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Start training')
    train_parser.add_argument('--type', '--trainer-type', 
                             choices=TrainerFactory.get_supported_trainers(),
                             help='Type of trainer to use')
    train_parser.add_argument('--model', '--model-name',
                             help='Name or path of the base model')
    train_parser.add_argument('--dataset', '--dataset-path',
                             help='Path to the training dataset')
    train_parser.add_argument('--output', '--output-dir',
                             default='./output',
                             help='Output directory for training results')
    train_parser.add_argument('--config', '--config-file',
                             help='Path to configuration file')
    
    # Common training arguments
    train_parser.add_argument('--learning-rate', type=float,
                             help='Learning rate for training')
    train_parser.add_argument('--epochs', '--num-epochs', type=int,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int,
                             help='Batch size per device')
    train_parser.add_argument('--max-length', type=int,
                             help='Maximum sequence length')
    
    # LoRA specific arguments
    train_parser.add_argument('--lora-r', type=int,
                             help='LoRA rank (r) parameter')
    train_parser.add_argument('--lora-alpha', type=int,
                             help='LoRA alpha parameter')
    train_parser.add_argument('--lora-dropout', type=float,
                             help='LoRA dropout rate')
    
    # DPO specific arguments
    train_parser.add_argument('--beta', type=float,
                             help='DPO beta parameter')
    
    # QLoRA specific arguments
    train_parser.add_argument('--use-4bit', action='store_true',
                             help='Use 4-bit quantization')
    train_parser.add_argument('--use-nested-quant', action='store_true',
                             help='Use nested quantization')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List supported trainer types')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get information about a trainer type')
    info_parser.add_argument('--type', '--trainer-type',
                            choices=TrainerFactory.get_supported_trainers(),
                            required=True,
                            help='Type of trainer to get info for')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a configuration file')
    validate_parser.add_argument('--config', '--config-file',
                                required=True,
                                help='Path to configuration file to validate')
    
    return parser


def train_from_args(args):
    """Create trainer and start training from command line arguments"""
    
    if args.config:
        # Load from configuration file
        print(f"Loading configuration from: {args.config}")
        trainer = TrainerFactory.create_trainer_from_config(args.config)
    else:
        # Create from command line arguments
        if not all([args.type, args.model, args.dataset]):
            print("Error: --type, --model, and --dataset are required when not using --config")
            sys.exit(1)
        
        # Build configuration from arguments
        config = {}
        
        # Common parameters
        if args.learning_rate:
            config['learning_rate'] = args.learning_rate
        if args.epochs:
            config['num_train_epochs'] = args.epochs
        if args.batch_size:
            config['per_device_train_batch_size'] = args.batch_size
        if args.max_length:
            config['max_seq_length'] = args.max_length
        
        # LoRA parameters
        if args.lora_r:
            config['lora_r'] = args.lora_r
        if args.lora_alpha:
            config['lora_alpha'] = args.lora_alpha
        if args.lora_dropout:
            config['lora_dropout'] = args.lora_dropout
        
        # DPO parameters
        if args.beta:
            config['beta'] = args.beta
        
        # QLoRA parameters
        if args.use_4bit:
            config['use_4bit'] = True
        if args.use_nested_quant:
            config['use_nested_quant'] = True
        
        print(f"Creating {args.type.upper()} trainer...")
        trainer = TrainerFactory.create_trainer(
            trainer_type=args.type,
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            **config
        )
    
    print("Starting training...")
    print(f"Configuration: {trainer.get_config()}")
    
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Evaluate the model
        print("Evaluating model...")
        results = trainer.evaluate()
        print(f"Evaluation results: {results}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


def list_trainers():
    """List all supported trainer types"""
    print("Supported Trainer Types:")
    print("=" * 50)
    
    for trainer_type in TrainerFactory.get_supported_trainers():
        info = TrainerFactory.get_trainer_info(trainer_type)
        print(f"\n{info['name']} ({trainer_type})")
        print(f"  Description: {info['description']}")
        print(f"  Class: {info['class']}")
        
        # Show key configuration parameters
        config = info['default_config']
        if config:
            print("  Key Parameters:")
            for key, value in list(config.items())[:5]:  # Show first 5
                print(f"    {key}: {value}")


def show_trainer_info(trainer_type):
    """Show detailed information about a specific trainer type"""
    try:
        info = TrainerFactory.get_trainer_info(trainer_type)
        
        print(f"Trainer Information: {info['name']}")
        print("=" * 50)
        print(f"Type: {trainer_type}")
        print(f"Class: {info['class']}")
        print(f"Description: {info['description']}")
        
        print("\nDefault Configuration:")
        print("-" * 30)
        for key, value in info['default_config'].items():
            print(f"  {key}: {value}")
        
        if info['supported_methods']:
            print(f"\nSupported Methods: {', '.join(info['supported_methods'])}")
            
    except Exception as e:
        print(f"Error getting trainer info: {e}")
        sys.exit(1)


def validate_config(config_file):
    """Validate a configuration file"""
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_file}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = ['trainer_type', 'model_name', 'dataset_path']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print(f"Error: Missing required fields: {missing_fields}")
            sys.exit(1)
        
        # Validate trainer type
        trainer_type = config['trainer_type']
        if trainer_type not in TrainerFactory.get_supported_trainers():
            print(f"Error: Unsupported trainer type: {trainer_type}")
            print(f"Supported types: {TrainerFactory.get_supported_trainers()}")
            sys.exit(1)
        
        # Validate configuration
        try:
            validated_config = TrainerFactory.validate_trainer_config(trainer_type, config)
            print(f"Configuration file '{config_file}' is valid!")
            print(f"Trainer type: {trainer_type}")
            print(f"Model: {config['model_name']}")
            print(f"Dataset: {config['dataset_path']}")
            print(f"Output directory: {config.get('output_dir', './output')}")
            
        except Exception as e:
            print(f"Error validating configuration: {e}")
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main CLI function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'train':
            train_from_args(args)
        elif args.command == 'list':
            list_trainers()
        elif args.command == 'info':
            show_trainer_info(args.type)
        elif args.command == 'validate':
            validate_config(args.config)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

