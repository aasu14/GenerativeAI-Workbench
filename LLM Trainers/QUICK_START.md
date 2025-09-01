# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd llm_trainers

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Basic Usage

```python
from llm_trainers import TrainerFactory

# Create an SFT trainer
trainer = TrainerFactory.create_trainer(
    trainer_type="sft",
    model_name="microsoft/DialoGPT-medium",
    dataset_path="your_data.json",
    output_dir="./output"
)

# Start training
trainer.train()
```

### 3. Command Line Usage

```bash
# List all supported trainer types
python -m llm_trainers.cli list

# Train with SFT
python -m llm_trainers.cli train \
    --type sft \
    --model microsoft/DialoGPT-medium \
    --dataset data.json \
    --output ./output

# Train with DPO
python -m llm_trainers.cli train \
    --type dpo \
    --model microsoft/DialoGPT-medium \
    --dataset preferences.json \
    --output ./output

# Train with QLoRA
python -m llm_trainers.cli train \
    --type qlora \
    --model microsoft/DialoGPT-medium \
    --dataset data.json \
    --output ./output \
    --lora-r 64
```

### 4. Run the Demo

```bash
# Run comprehensive demo
python -m llm_trainers.demo

# Run basic examples
python -m llm_trainers.examples.basic_usage
```

## 📊 Supported Trainer Types

| Trainer | Type | Description | Use Case |
|---------|------|-------------|----------|
| **SFT** | Supervised Fine-Tuning | Standard instruction-following training | General fine-tuning |
| **DPO** | Direct Preference Optimization | Preference learning | Alignment, safety |
| **RLHF** | Reinforcement Learning from Human Feedback | Policy optimization | Advanced alignment |
| **QLoRA** | Quantized LoRA | Efficient 4-bit training | Memory-constrained |
| **PEFT** | Parameter-Efficient Fine-Tuning | Multiple PEFT methods | Various efficiency needs |
| **LoRA** | Low-Rank Adaptation | Standard LoRA | Balanced efficiency |
| **AdaLoRA** | Adaptive LoRA | Dynamic rank allocation | Optimal efficiency |

## 🔧 Data Formats

### SFT/QLoRA/LoRA Data
```json
[
    {
        "instruction": "Your instruction here",
        "output": "Expected response here"
    }
]
```

### DPO Data
```json
[
    {
        "prompt": "Your prompt here",
        "chosen": "Preferred response",
        "rejected": "Dispreferred response"
    }
]
```

### RLHF Data
```json
[
    {
        "query": "Your prompt here"
    }
]
```

## ⚡ Quick Examples

### SFT Training
```python
trainer = TrainerFactory.create_trainer(
    trainer_type="sft",
    model_name="microsoft/DialoGPT-medium",
    dataset_path="data.json",
    output_dir="./output",
    learning_rate=5e-5,
    num_train_epochs=3
)
trainer.train()
```

### DPO Training
```python
trainer = TrainerFactory.create_trainer(
    trainer_type="dpo",
    model_name="microsoft/DialoGPT-medium",
    dataset_path="preferences.json",
    output_dir="./output",
    beta=0.1,
    use_peft=True
)
trainer.train()
```

### QLoRA Training
```python
trainer = TrainerFactory.create_trainer(
    trainer_type="qlora",
    model_name="microsoft/DialoGPT-medium",
    dataset_path="data.json",
    output_dir="./output",
    lora_r=64,
    use_4bit=True
)
trainer.train()
```

## 🎯 Next Steps

1. **Run the demo**: `python -m llm_trainers.demo`
2. **Check examples**: `python -m llm_trainers.examples.basic_usage`
3. **Explore CLI**: `python -m llm_trainers.cli --help`
4. **Read documentation**: Check the README.md for detailed information
5. **Customize**: Modify configurations for your specific use case

## 🆘 Need Help?

- Check the examples in `examples/` directory
- Run `python -m llm_trainers.cli info --type <trainer_type>`
- Review the test files for usage patterns
- Check the comprehensive demo script

## 🚨 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM (16GB+ recommended)
- Hugging Face Transformers 4.35+
- TRL 0.7+
- PEFT 0.6+

