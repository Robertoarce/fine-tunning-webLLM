"""
RAFT Fine-Tuning Script
Fine-tune a language model using RAFT (Retrieval Augmented Fine Tuning) methodology.
"""

import os
import json
import torch
import logging
import yaml
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
from torch.utils.data import random_split
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAFTTrainer:
    """RAFT Fine-tuning trainer"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize RAFT trainer with configuration"""
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        model_name = self.config['model']['name']
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        model_kwargs = {
            'torch_dtype': torch.float16 if self.config['model']['fp16'] else torch.float32,
        }
        
        if torch.cuda.is_available():
            model_kwargs['device_map'] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Resize token embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Model and tokenizer loaded successfully")
    
    def load_raft_data(self, data_path: str) -> List[Dict]:
        """Load RAFT training data from JSONL file"""
        logger.info(f"Loading RAFT data from {data_path}")
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} training examples")
        return data
    
    def format_raft_example(self, example: Dict) -> str:
        """
        Format a RAFT example into a training string
        
        The format follows instruction-tuning style:
        Instruction + Output
        """
        # Combine instruction and output for causal LM training
        formatted = f"{example['instruction']}\n{example['output']}{self.tokenizer.eos_token}"
        return formatted
    
    def prepare_datasets(self, data_path: str = "raft_training_data.jsonl"):
        """Prepare training and validation datasets"""
        logger.info("Preparing RAFT datasets...")
        
        # Load RAFT data
        raft_data = self.load_raft_data(data_path)
        
        # Format examples
        formatted_texts = [self.format_raft_example(ex) for ex in raft_data]
        
        # Tokenize
        logger.info("Tokenizing examples...")
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            max_length=self.config['model']['max_length'],
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()
        })
        
        # Split into train and validation
        train_split = self.config['data']['train_split']
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        # Print example
        logger.info("\nExample training instance:")
        logger.info(formatted_texts[0][:500] + "...")
    
    def setup_training_args(self) -> tuple:
        """Setup training arguments"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config['output']['base_dir']}_raft_{timestamp}"
        
        training_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=training_config['num_epochs'],
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_steps=training_config['warmup_steps'],
            fp16=self.config['model']['fp16'],
            logging_steps=training_config['logging_steps'],
            eval_steps=training_config['eval_steps'],
            save_steps=training_config['save_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=self.config['output']['save_total_limit'],
            prediction_loss_only=True,
            remove_unused_columns=True,
            dataloader_pin_memory=training_config['dataloader_pin_memory'],
            dataloader_num_workers=training_config['dataloader_num_workers'],
            report_to="wandb" if self.config['logging']['use_wandb'] else "none",
        )
        
        return training_args, output_dir
    
    def train(self):
        """Execute RAFT fine-tuning"""
        logger.info("Starting RAFT fine-tuning process...")
        
        # Initialize W&B if enabled
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                config=self.config,
                name=f"raft-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Prepare datasets
        self.prepare_datasets()
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Setup training arguments
        training_args, output_dir = self.setup_training_args()
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['training']['early_stopping_patience']
                )
            ]
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training logs
        with open(f"{output_dir}/training_log.json", 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)
        
        logger.info(f"Training completed! Model saved to: {output_dir}")
        
        # Test the model
        self.test_model(output_dir)
        
        return output_dir
    
    def test_model(self, model_path: str = None):
        """Test the RAFT fine-tuned model"""
        logger.info("\n" + "="*80)
        logger.info("Testing RAFT fine-tuned model...")
        logger.info("="*80)
        
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test with RAFT-style prompts
        test_prompts = [
            """Use the following documents to answer the question.

Document [1]:
Roberto Arce is a Data Scientist and Machine Learning Engineer based in France. He works at Sanofi providing data science services with a focus on machine learning solutions and data analysis.

Question: Where does Roberto work and what does he do?
Answer:""",
            """Use the following documents to answer the question.

Document [1]:
Roberto has expertise in Python, JavaScript, Vue.js, and SQL. He's proficient in machine learning frameworks like scikit-learn, TensorFlow, PyTorch, Pandas, and NumPy.

Question: What experience does Roberto have with Python?
Answer:""",
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n--- Test {i} ---")
            logger.info(f"Prompt:\n{prompt[:200]}...")
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                self.model = self.model.to('cuda')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.95
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the answer part
            answer = generated_text[len(prompt):].strip()
            logger.info(f"Generated Answer:\n{answer}\n")


def main():
    """Main training function"""
    # Initialize RAFT trainer
    trainer = RAFTTrainer(config_path="config.yaml")
    
    # Train the model
    output_dir = trainer.train()
    
    logger.info("\n" + "="*80)
    logger.info("RAFT Fine-tuning Complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

