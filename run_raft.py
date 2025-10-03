"""
Complete RAFT Pipeline Runner
Generates RAFT dataset and fine-tunes model in one script.
"""

import logging
import sys
from pathlib import Path

from raft_data_generator import RAFTDataGenerator
from raft_fine_tuning import RAFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete RAFT pipeline"""
    
    logger.info("="*80)
    logger.info("RAFT (Retrieval Augmented Fine Tuning) Pipeline")
    logger.info("="*80)
    
    # Step 1: Generate RAFT dataset
    logger.info("\n[STEP 1] Generating RAFT Training Dataset...")
    logger.info("-"*80)
    
    try:
        generator = RAFTDataGenerator(
            source_file="roberto_data.txt",
            output_file="raft_training_data.jsonl",
            chunk_size=300,
            num_distractors=3,
            distractor_probability=0.5
        )
        
        dataset_path = generator.generate_and_save()
        logger.info(f"✓ RAFT dataset generated successfully: {dataset_path}")
        
    except Exception as e:
        logger.error(f"✗ Failed to generate RAFT dataset: {e}")
        sys.exit(1)
    
    # Step 2: Fine-tune model with RAFT
    logger.info("\n[STEP 2] Fine-tuning Model with RAFT...")
    logger.info("-"*80)
    
    try:
        trainer = RAFTTrainer(config_path="config.yaml")
        model_path = trainer.train()
        logger.info(f"✓ Model fine-tuned successfully: {model_path}")
        
    except Exception as e:
        logger.error(f"✗ Failed to fine-tune model: {e}")
        sys.exit(1)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("RAFT Pipeline Complete! 🎉")
    logger.info("="*80)
    logger.info(f"Training Data: {dataset_path}")
    logger.info(f"Fine-tuned Model: {model_path}")
    logger.info("\nYou can now use the model for retrieval-augmented question answering!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

