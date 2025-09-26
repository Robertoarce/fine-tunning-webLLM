# Fine-tuning 

This project fine-tunes a GPT-2 model on my professional data to create a personalized language model.

## Files Overview

- `fine-tunning.py` - Basic fine-tuning script (original, there is an advanced too, but this is working well)
- `fine_tuning_lora.py` - LoRA fine-tuning script (memory efficient)
- `config.yaml` - Configuration file for training parameters
- `requirements.txt` - Python dependencies
- `roberto_data.txt` - Training data 

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run basic fine-tuning:**

   ```bash
   python fine-tunning.py
   ```

3. **Run LoRA fine-tuning (memory efficient):**
   ```bash
   python fine_tuning_lora.py
   ```
 

## Configuration Options

Edit `config.yaml` to customize:

- **Model**: Currently `distilgpt2`, `gpt2`, `gpt2-medium`, `gpt2-large`
- **Training**: Epochs, batch size, learning rate, etc.
- **Data**: File path, train/validation split
- **Output**: Save directory, checkpoint limits
- **Logging**: Log level, Weights & Biases integration
- **LoRA**: Rank, alpha, target modules, dropout, quantization
 
## Model Testing

After training, the model can generate text about Roberto Arce:

- "Roberto Arce is a Data Scientist and Machine Learning Engineer..."
- "Roberto's expertise includes machine learning, statistical analysis..."
- "Roberto studied Industrial Engineering and has multiple master's degrees..."
 
