# LLM Trainers Repository

A comprehensive collection of LLM fine-tuning trainers including SFT, DPO, RLHF, and more.

## Features

- **SFT Trainer**: Supervised Fine-Tuning for instruction following
- **DPO Trainer**: Direct Preference Optimization for alignment
- **RLHF Trainer**: Reinforcement Learning from Human Feedback
- **QLoRA Trainer**: Quantized Low-Rank Adaptation for efficient fine-tuning
- **PEFT Trainer**: Parameter-Efficient Fine-Tuning
- **LoRA Trainer**: Low-Rank Adaptation
- **AdaLoRA Trainer**: Adaptive Low-Rank Adaptation

## Quick Start

```python
from llm_trainers import TrainerFactory

# Initialize trainer
trainer = TrainerFactory.create_trainer(
    trainer_type="sft",
    model_name="microsoft/DialoGPT-medium",
    dataset_path="your_dataset.json",
    output_dir="./output"
)

# Start training
trainer.train()
```

## Installation

```bash
pip install -r requirements.txt
```

## Supported Trainer Types

- `sft` - Supervised Fine-Tuning
- `dpo` - Direct Preference Optimization
- `rlhf` - Reinforcement Learning from Human Feedback
- `qlora` - Quantized LoRA
- `peft` - Parameter-Efficient Fine-Tuning
- `lora` - Low-Rank Adaptation
- `adalora` - Adaptive LoRA

## Data Format

Each trainer expects data in specific formats. See individual trainer documentation for details.

