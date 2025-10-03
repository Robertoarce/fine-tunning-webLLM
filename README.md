# Fine-tuning 

This project fine-tunes language models on professional data using multiple approaches including traditional fine-tuning, LoRA, and **RAFT (Retrieval Augmented Fine Tuning)**.

## Files Overview

### Traditional Fine-tuning
- `fine-tunning.py` - Basic fine-tuning script (original, working well)
- `fine_tuning_lora.py` - LoRA fine-tuning script (memory efficient)

### RAFT (Retrieval Augmented Fine Tuning) 🆕
- `raft_data_generator.py` - Generates RAFT-formatted training data with context documents
- `raft_fine_tuning.py` - RAFT fine-tuning implementation
- `run_raft.py` - Complete RAFT pipeline (data generation + training)
- `test_raft_model.py` - Test and interact with RAFT models

### Configuration & Data
- `config.yaml` - Configuration file for training parameters
- `requirements.txt` - Python dependencies
- `roberto_data.txt` - Source training data

## What is RAFT?

**RAFT (Retrieval Augmented Fine Tuning)** is an advanced fine-tuning technique that trains models to answer questions using provided context documents. Unlike traditional fine-tuning, RAFT:

- ✅ Trains the model to extract answers from relevant documents
- ✅ Includes "distractor" documents to improve robustness
- ✅ Makes models better at retrieval-augmented generation (RAG)
- ✅ Reduces hallucination by grounding answers in context

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose Your Approach

#### Option A: RAFT Fine-tuning (Recommended for Q&A) 🌟

Run the complete RAFT pipeline:

```bash
python run_raft.py
```

This will:
1. Generate RAFT training data from `roberto_data.txt`
2. Fine-tune the model with retrieval-augmented examples
3. Test the model with sample queries

Or run steps separately:

```bash
# Step 1: Generate RAFT dataset
python raft_data_generator.py

# Step 2: Fine-tune with RAFT
python raft_fine_tuning.py

# Step 3: Test the model
python test_raft_model.py
```

#### Option B: Traditional Fine-tuning

```bash
python fine-tunning.py
```

#### Option C: LoRA Fine-tuning (Memory Efficient)

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

### RAFT-Specific Configuration

When using RAFT, you can customize in `raft_data_generator.py`:

```python
RAFTDataGenerator(
    source_file="roberto_data.txt",
    output_file="raft_training_data.jsonl",
    chunk_size=300,              # Size of document chunks
    num_distractors=3,           # Number of distractor docs per example
    distractor_probability=0.5   # 50% of examples include distractors
)
```

## Model Testing

### Traditional Model Testing

After training, the model can generate text about Roberto Arce:

- "Roberto Arce is a Data Scientist and Machine Learning Engineer..."
- "Roberto's expertise includes machine learning, statistical analysis..."
- "Roberto studied Industrial Engineering and has multiple master's degrees..."

### RAFT Model Testing

Test the RAFT model with document-based Q&A:

```bash
python test_raft_model.py [model_path]
```

Example RAFT query format:
```
Question: Where does Roberto work?
Documents:
- Roberto Arce works at Sanofi as a Data Scientist since 2023
- He specializes in machine learning solutions

Answer: Based on the documents, Roberto works at Sanofi as a Data Scientist.
```

## RAFT Training Data Format

RAFT generates training examples in this format:

```json
{
  "instruction": "Use the following documents to answer...\nDocument [1]: ...\nQuestion: ...",
  "output": "Based on the documents: ...",
  "oracle_context": "The relevant document containing the answer",
  "num_distractors": 3,
  "has_distractors": true
}
```

## Comparison: Traditional vs RAFT

| Feature | Traditional | RAFT |
|---------|-------------|------|
| Training Style | Completion | Question + Context → Answer |
| Use Case | Text generation | Question answering with documents |
| Hallucination | Higher risk | Lower (grounded in context) |
| RAG Integration | Separate step | Built-in during training |
| Best For | Creative writing | Factual Q&A |
 
