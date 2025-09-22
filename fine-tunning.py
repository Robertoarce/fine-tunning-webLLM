import os
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
    EarlyStoppingCallback
)
from torch.utils.data import random_split
import wandb
from sklearn.metrics import accuracy_score
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "model_name": "distilgpt2",
    "data_file": "roberto_data.txt",
    "output_dir": "./finetuned_roberto",
    "max_length": 512,  # Increased from 128
    "train_split": 0.9,  # 90% for training, 10% for validation
    "num_epochs": 5,  # Increased from 3
    "batch_size": 8,  # Increased from 4
    "learning_rate": 5e-5,  # Added learning rate
    "warmup_steps": 100,  # Added warmup
    "weight_decay": 0.01,  # Added weight decay
    "gradient_accumulation_steps": 2,  # Added gradient accumulation
    "fp16": True,  # Enable mixed precision training
    "use_wandb": False,  # Set to True if you want to use Weights & Biases
}


def setup_model_and_tokenizer():
    """Initialize model and tokenizer with proper configuration"""
    logger.info(f"Loading model: {CONFIG['model_name']}")

    # Load tokenizer with padding token
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.float16 if CONFIG['fp16'] else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def prepare_datasets(tokenizer):
    """Prepare training and validation datasets"""
    logger.info("Preparing datasets...")

    # Load the full dataset
    full_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=CONFIG['data_file'],
        block_size=CONFIG['max_length'],
    )

    # Split into train and validation
    train_size = int(CONFIG['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    # For language modeling, we typically use perplexity
    # This is a simplified version - you might want to implement proper perplexity calculation
    predictions = np.argmax(predictions, axis=-1)

    # Mask out padding tokens
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


def setup_training_args():
    """Setup training arguments with best practices"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{CONFIG['output_dir']}_{timestamp}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_steps=CONFIG['warmup_steps'],
        fp16=CONFIG['fp16'],
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        prediction_loss_only=False,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to="wandb" if CONFIG['use_wandb'] else None,
    )

    return training_args, output_dir


def main():
    """Main training function"""
    logger.info("Starting fine-tuning process...")

    # Initialize Weights & Biases if enabled
    if CONFIG['use_wandb']:
        wandb.init(project="roberto-finetuning", config=CONFIG)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(tokenizer)

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Optimize for GPU memory
    )

    # Setup training arguments
    training_args, output_dir = setup_training_args()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training logs
    trainer.log_metrics("train", trainer.state.log_history[-1])

    logger.info(f"Training completed! Model saved to: {output_dir}")

    # Test the model with a sample prompt
    test_model(model, tokenizer)


def test_model(model, tokenizer):
    """Test the fine-tuned model with a sample prompt"""
    logger.info("Testing the fine-tuned model...")

    test_prompt = "Roberto Arce is a"

    # Tokenize input
    inputs = tokenizer.encode(test_prompt, return_tensors="pt")

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()
