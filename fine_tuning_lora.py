import os
import yaml
import torch
import logging
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from torch.utils.data import random_split
import wandb
from sklearn.metrics import accuracy_score
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class RobertoLoRAFineTuner:
    """LoRA fine-tuning class for Roberto Arce model"""

    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config['logging']['level'])
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.peft_model = None

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA configuration"""
        model_config = self.config['model']
        lora_config = self.config['lora']

        self.logger.info(f"Loading model: {model_config['name']}")

        # Load tokenizer with padding token
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup quantization config if enabled
        quantization_config = None
        if model_config.get('load_in_4bit', False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if model_config['fp16'] else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif model_config.get('load_in_8bit', False):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if model_config['fp16'] else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Setup LoRA configuration
        if lora_config['enabled']:
            self.logger.info("Setting up LoRA configuration...")

            # Get target modules for the specific model
            target_modules = lora_config.get('target_modules')
            if target_modules is None or target_modules == ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                # Use default target modules for the model
                model_name = model_config['name'].lower()
                if 'gpt2' in model_name or 'distilgpt2' in model_name:
                    target_modules = ["c_attn", "c_proj"]
                else:
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

            peft_config = LoraConfig(
                r=lora_config['r'],
                lora_alpha=lora_config['lora_alpha'],
                target_modules=target_modules,
                lora_dropout=lora_config['lora_dropout'],
                bias=lora_config['bias'],
                task_type=TaskType.CAUSAL_LM,
            )

            # Apply LoRA to the model
            self.peft_model = get_peft_model(self.model, peft_config)
            self.peft_model.print_trainable_parameters()

            self.logger.info(
                f"LoRA applied! Trainable parameters: {self.peft_model.num_parameters()}")
        else:
            self.peft_model = self.model
            self.logger.info("LoRA disabled - using full fine-tuning")

    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        self.logger.info("Preparing datasets...")

        data_config = self.config['data']

        # Load the full dataset
        full_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=data_config['file_path'],
            block_size=self.config['model']['max_length'],
        )

        # Split into train and validation
        train_split = data_config['train_split']
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)

        # Mask out padding tokens
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]

        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    def setup_training_args(self):
        """Setup training arguments optimized for LoRA"""
        training_config = self.config['training']
        output_config = self.config['output']

        # Create output directory with timestamp if enabled
        if output_config['include_timestamp']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{output_config['base_dir']}_lora_{timestamp}"
        else:
            output_dir = f"{output_config['base_dir']}_lora"

        # LoRA-specific training arguments
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
            save_total_limit=output_config['save_total_limit'],
            prediction_loss_only=False,
            remove_unused_columns=False,
            dataloader_pin_memory=training_config['dataloader_pin_memory'],
            dataloader_num_workers=training_config['dataloader_num_workers'],
            report_to="wandb" if self.config['logging']['use_wandb'] else None,
            # LoRA-specific optimizations
            gradient_checkpointing=True,  # Save memory
            optim="adamw_torch",  # Better optimizer for LoRA
            lr_scheduler_type="cosine",  # Better learning rate schedule
        )

        return training_args, output_dir

    def train(self):
        """Main training function with LoRA"""
        self.logger.info("Starting LoRA fine-tuning process...")

        # Initialize Weights & Biases if enabled
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=f"{self.config['logging']['wandb_project']}-lora",
                config=self.config
            )

        # Setup model and tokenizer
        self.setup_model_and_tokenizer()

        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets()

        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Optimize for GPU memory
        )

        # Setup training arguments
        training_args, output_dir = self.setup_training_args()

        # Initialize trainer
        self.trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config['training']['early_stopping_patience']
            )]
        )

        # Start training
        self.logger.info("Starting LoRA training...")
        self.trainer.train()

        # Save the final model
        self.logger.info("Saving final LoRA model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # Save LoRA adapters separately
        if self.config['lora']['enabled']:
            self.peft_model.save_pretrained(f"{output_dir}/lora_adapters")
            self.logger.info(
                f"LoRA adapters saved to: {output_dir}/lora_adapters")

        # Save training logs
        self.trainer.log_metrics("train", self.trainer.state.log_history[-1])

        self.logger.info(
            f"LoRA training completed! Model saved to: {output_dir}")

        return output_dir

    def test_model(self, prompt="Roberto Arce is a"):
        """Test the fine-tuned model with a sample prompt"""
        if self.peft_model is None or self.tokenizer is None:
            self.logger.error("Model not loaded. Please run train() first.")
            return

        self.logger.info("Testing the LoRA fine-tuned model...")

        generation_config = self.config['generation']

        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate text
        with torch.no_grad():
            outputs = self.peft_model.generate(
                inputs,
                max_length=generation_config['max_length'],
                num_return_sequences=generation_config['num_return_sequences'],
                temperature=generation_config['temperature'],
                do_sample=generation_config['do_sample'],
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and print
        generated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        self.logger.info(f"Generated text: {generated_text}")

        return generated_text

    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """Generate text with custom parameters"""
        if self.peft_model is None or self.tokenizer is None:
            self.logger.error("Model not loaded. Please run train() first.")
            return

        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.peft_model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def load_lora_model(self, model_path):
        """Load a previously trained LoRA model"""
        self.logger.info(f"Loading LoRA model from: {model_path}")

        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            torch_dtype=torch.float16 if self.config['model']['fp16'] else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Load LoRA adapters
        if os.path.exists(f"{model_path}/lora_adapters"):
            self.peft_model = PeftModel.from_pretrained(
                self.model, f"{model_path}/lora_adapters")
            self.logger.info("LoRA adapters loaded successfully!")
        else:
            self.peft_model = self.model
            self.logger.warning("No LoRA adapters found, using base model")


def main():
    """Main function to run the LoRA fine-tuning process"""
    # Initialize the LoRA fine-tuner
    fine_tuner = RobertoLoRAFineTuner()

    # Train the model
    fine_tuner.train()

    # Test the model
    fine_tuner.test_model()

    # Additional test prompts
    test_prompts = [
        "Roberto Arce works as a",
        "Roberto's expertise includes",
        "Roberto studied",
        "Roberto's professional experience"
    ]

    print("\n" + "="*50)
    print("TESTING LoRA MODEL WITH DIFFERENT PROMPTS")
    print("="*50)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {fine_tuner.generate_text(prompt)}")


if __name__ == "__main__":
    main()
