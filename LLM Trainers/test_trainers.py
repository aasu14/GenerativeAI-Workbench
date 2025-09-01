"""
Test file for LLM Trainers
Tests all trainer types to ensure they work correctly
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
from pathlib import Path

from .trainer_factory import TrainerFactory
from .trainers.sft_trainer import SFTTrainer
from .trainers.dpo_trainer import DPOTrainer
from .trainers.rlhf_trainer import RLHFTrainer
from .trainers.qlora_trainer import QLoRATrainer
from .trainers.peft_trainer import PEFTTrainer
from .trainers.lora_trainer import LoRATrainer
from .trainers.adalora_trainer import AdaLoRATrainer


class TestTrainerFactory(unittest.TestCase):
    """Test cases for TrainerFactory"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset.json")
        
        # Create a test dataset
        test_data = [
            {"instruction": "Test instruction 1", "output": "Test output 1"},
            {"instruction": "Test instruction 2", "output": "Test output 2"}
        ]
        
        with open(self.dataset_path, 'w') as f:
            json.dump(test_data, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_supported_trainers(self):
        """Test getting supported trainer types"""
        supported = TrainerFactory.get_supported_trainers()
        expected = ["sft", "dpo", "rlhf", "qlora", "peft", "lora", "adalora"]
        
        self.assertEqual(set(supported), set(expected))
    
    def test_create_sft_trainer(self):
        """Test creating SFT trainer"""
        with patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.AutoModelForCausalLM.from_pretrained'):
            
            trainer = TrainerFactory.create_trainer(
                trainer_type="sft",
                model_name="test-model",
                dataset_path=self.dataset_path,
                output_dir=self.temp_dir
            )
            
            self.assertIsInstance(trainer, SFTTrainer)
    
    def test_create_dpo_trainer(self):
        """Test creating DPO trainer"""
        with patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.AutoModelForCausalLM.from_pretrained'):
            
            trainer = TrainerFactory.create_trainer(
                trainer_type="dpo",
                model_name="test-model",
                dataset_path=self.dataset_path,
                output_dir=self.temp_dir
            )
            
            self.assertIsInstance(trainer, DPOTrainer)
    
    def test_create_qlora_trainer(self):
        """Test creating QLoRA trainer"""
        with patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.AutoModelForCausalLM.from_pretrained'):
            
            trainer = TrainerFactory.create_trainer(
                trainer_type="qlora",
                model_name="test-model",
                dataset_path=self.dataset_path,
                output_dir=self.temp_dir
            )
            
            self.assertIsInstance(trainer, QLoRATrainer)
    
    def test_invalid_trainer_type(self):
        """Test creating trainer with invalid type"""
        with self.assertRaises(ValueError):
            TrainerFactory.create_trainer(
                trainer_type="invalid",
                model_name="test-model",
                dataset_path=self.dataset_path,
                output_dir=self.temp_dir
            )
    
    def test_get_trainer_info(self):
        """Test getting trainer information"""
        info = TrainerFactory.get_trainer_info("sft")
        
        self.assertIn("name", info)
        self.assertIn("class", info)
        self.assertIn("description", info)
        self.assertIn("default_config", info)
        self.assertEqual(info["name"], "SFT")


class TestSFTTrainer(unittest.TestCase):
    """Test cases for SFTTrainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset.json")
        
        # Create a test dataset
        test_data = [
            {"instruction": "Test instruction 1", "output": "Test output 1"},
            {"instruction": "Test instruction 2", "output": "Test output 2"}
        ]
        
        with open(self.dataset_path, 'w') as f:
            json.dump(test_data, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_sft_trainer_initialization(self, mock_model, mock_tokenizer):
        """Test SFT trainer initialization"""
        # Mock the model and tokenizer
        mock_tokenizer.return_value.pad_token = None
        mock_model.return_value = Mock()
        
        trainer = SFTTrainer(
            model_name="test-model",
            dataset_path=self.dataset_path,
            output_dir=self.temp_dir
        )
        
        self.assertEqual(trainer.model_name, "test-model")
        self.assertEqual(trainer.dataset_path, self.dataset_path)
        self.assertEqual(trainer.max_seq_length, 512)
        self.assertFalse(trainer.packing)
    
    def test_sft_default_config(self):
        """Test SFT default configuration"""
        with patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.AutoModelForCausalLM.from_pretrained'):
            
            trainer = SFTTrainer(
                model_name="test-model",
                dataset_path=self.dataset_path,
                output_dir=self.temp_dir
            )
            
            config = trainer.get_config()
            self.assertIn("learning_rate", config)
            self.assertIn("num_train_epochs", config)
            self.assertEqual(config["learning_rate"], 5e-5)


class TestDPOTrainer(unittest.TestCase):
    """Test cases for DPOTrainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset.json")
        
        # Create a test preference dataset
        test_data = [
            {"prompt": "Test prompt 1", "chosen": "Good response", "rejected": "Bad response"},
            {"prompt": "Test prompt 2", "chosen": "Better response", "rejected": "Worse response"}
        ]
        
        with open(self.dataset_path, 'w') as f:
            json.dump(test_data, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_dpo_trainer_initialization(self, mock_model, mock_tokenizer):
        """Test DPO trainer initialization"""
        # Mock the model and tokenizer
        mock_tokenizer.return_value.pad_token = None
        mock_model.return_value = Mock()
        
        trainer = DPOTrainer(
            model_name="test-model",
            dataset_path=self.dataset_path,
            output_dir=self.temp_dir
        )
        
        self.assertEqual(trainer.beta, 0.1)
        self.assertEqual(trainer.max_prompt_length, 512)
        self.assertEqual(trainer.max_length, 1024)
        self.assertTrue(trainer.use_peft)


class TestQLoRATrainer(unittest.TestCase):
    """Test cases for QLoRATrainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset.json")
        
        # Create a test dataset
        test_data = [
            {"instruction": "Test instruction 1", "output": "Test output 1"},
            {"instruction": "Test instruction 2", "output": "Test output 2"}
        ]
        
        with open(self.dataset_path, 'w') as f:
            json.dump(test_data, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_qlora_trainer_initialization(self, mock_model, mock_tokenizer):
        """Test QLoRA trainer initialization"""
        # Mock the model and tokenizer
        mock_tokenizer.return_value.pad_token = None
        mock_model.return_value = Mock()
        
        trainer = QLoRATrainer(
            model_name="test-model",
            dataset_path=self.dataset_path,
            output_dir=self.temp_dir
        )
        
        self.assertEqual(trainer.lora_r, 64)
        self.assertEqual(trainer.lora_alpha, 16)
        self.assertEqual(trainer.lora_dropout, 0.1)
        self.assertTrue(trainer.use_4bit)
        self.assertTrue(trainer.use_nested_quant)
    
    def test_qlora_lora_config(self):
        """Test QLoRA LoRA configuration"""
        with patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.AutoModelForCausalLM.from_pretrained'):
            
            trainer = QLoRATrainer(
                model_name="test-model",
                dataset_path=self.dataset_path,
                output_dir=self.temp_dir
            )
            
            lora_config = trainer.get_lora_config()
            self.assertIn("lora_r", lora_config)
            self.assertIn("lora_alpha", lora_config)
            self.assertIn("use_4bit", lora_config)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

