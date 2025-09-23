# Fine-tuning 

This project fine-tunes a GPT-2 model on my professional data to create a personalized language model.

## Files Overview

- `fine-tunning.py` - Basic fine-tuning script (original)
- `fine_tuning_advanced.py` - Enhanced fine-tuning script with better practices
- `fine_tuning_lora.py` - LoRA fine-tuning script (memory efficient)
- `config.yaml` - Configuration file for training parameters
- `requirements.txt` - Python dependencies
- `roberto_data.txt` - Training data (Roberto's professional information)

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run basic fine-tuning:**

   ```bash
   python fine-tunning.py
   ```

3. **Run advanced fine-tuning (recommended):**

   ```bash
   python fine_tuning_advanced.py
   ```

4. **Run LoRA fine-tuning (memory efficient):**
   ```bash
   python fine_tuning_lora.py
   ```

## Key Improvements Made

### 1. **Better Data Handling**

- Train/validation split (90%/10%)
- Proper padding token handling
- Increased context length (512 vs 128 tokens)

### 2. **Enhanced Training Configuration**

- Learning rate scheduling with warmup
- Gradient accumulation for larger effective batch sizes
- Mixed precision training (FP16) for memory efficiency
- Early stopping to prevent overfitting
- Better evaluation metrics

### 3. **Memory Optimization**

- Automatic device mapping
- Optimized data collator
- Gradient accumulation
- Mixed precision training

### 4. **Monitoring & Logging**

- Comprehensive logging
- Optional Weights & Biases integration
- Training metrics tracking
- Model testing functionality

### 5. **Configuration Management**

- YAML-based configuration
- Easy parameter tuning
- Reproducible experiments

### 6. **LoRA (Low-Rank Adaptation) Support**

- Memory-efficient fine-tuning
- Faster training with less GPU memory
- Better performance with limited data
- Quantization support (4-bit/8-bit)
- LoRA adapters can be saved and loaded separately

## Configuration Options

Edit `config.yaml` to customize:

- **Model**: Choose between `distilgpt2`, `gpt2`, `gpt2-medium`, `gpt2-large`
- **Training**: Epochs, batch size, learning rate, etc.
- **Data**: File path, train/validation split
- **Output**: Save directory, checkpoint limits
- **Logging**: Log level, Weights & Biases integration
- **LoRA**: Rank, alpha, target modules, dropout, quantization

## Usage Examples

### Basic Usage

```python
from fine_tuning_advanced import RobertoFineTuner

# Initialize and train
fine_tuner = RobertoFineTuner()
fine_tuner.train()

# Generate text
text = fine_tuner.generate_text("Roberto Arce is a", max_length=100)
print(text)
```

### Custom Configuration

```python
# Use custom config file
fine_tuner = RobertoFineTuner("my_config.yaml")
fine_tuner.train()
```

### LoRA Fine-tuning

```python
from fine_tuning_lora import RobertoLoRAFineTuner

# Initialize LoRA fine-tuner
lora_fine_tuner = RobertoLoRAFineTuner()
lora_fine_tuner.train()

# Generate text
text = lora_fine_tuner.generate_text("Roberto Arce is a", max_length=100)
print(text)
```

## Model Testing

After training, the model can generate text about Roberto Arce:

- "Roberto Arce is a Data Scientist and Machine Learning Engineer..."
- "Roberto's expertise includes machine learning, statistical analysis..."
- "Roberto studied Industrial Engineering and has multiple master's degrees..."

## Hardware Requirements

### Full Fine-tuning

- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB+ RAM, GPU with 8GB+ VRAM
- **Optimal**: GPU with 16GB+ VRAM for larger models

### LoRA Fine-tuning

- **Minimum**: 4GB RAM, CPU training
- **Recommended**: 8GB+ RAM, GPU with 4GB+ VRAM
- **Optimal**: GPU with 8GB+ VRAM for larger models
- **Memory Savings**: 50-90% less memory usage compared to full fine-tuning

## Tips for Better Results

1. **More Data**: Add more diverse text about Roberto
2. **Longer Training**: Increase epochs if validation loss is still decreasing
3. **Larger Model**: Use `gpt2-medium` or `gpt2-large` for better quality
4. **Data Quality**: Ensure consistent formatting and style
5. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or use gradient accumulation
- **Slow training**: Enable FP16 and increase batch size
- **Poor quality**: Check data format and increase training epochs
- **Overfitting**: Reduce learning rate or add more regularization
