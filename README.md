# RAFT Fine-Tuning Project

**Retrieval Augmented Fine Tuning (RAFT)** - Fine-tune language models to answer questions using context documents with reduced hallucination.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What is RAFT?

RAFT is an advanced fine-tuning technique that trains language models to:

- ✅ **Answer questions using provided context documents**
- ✅ **Distinguish between relevant and irrelevant information**
- ✅ **Reduce hallucination by grounding answers in documents**
- ✅ **Work seamlessly with RAG (Retrieval Augmented Generation) systems**

Unlike traditional fine-tuning that teaches models to generate text, RAFT teaches models to **extract and synthesize information from context**.

---

## 🚀 Quick Start

### Option A: Complete RAFT + RAG System (Recommended) 🌟

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train RAFT model
python run_raft.py

# 3. Run RAG system with RAFT model
python rag_system.py
```

**Result:** A complete question-answering system with document retrieval and RAFT-powered generation!

### Option B: RAFT Only (Training & Testing)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run RAFT pipeline
python run_raft.py

# 3. Test RAFT model
python test_raft_model.py
```

### Option C: RAG System Only (No Training)

```bash
# 1. Install RAG dependencies
pip install chromadb sentence-transformers

# 2. Run RAG system
python rag_system.py
```

**Note:** Without RAFT model, you'll need to provide a generation model or use base model.

---

## 📁 Project Structure

```
.
├── raft_data_generator.py      # Generates RAFT training dataset
├── raft_fine_tuning.py         # RAFT fine-tuning implementation
├── run_raft.py                 # Complete pipeline (data + training)
├── test_raft_model.py          # Test and interact with models
├── example_raft_usage.py       # Interactive demo/tutorial
│
├── rag_system.py               # 🆕 Complete RAG implementation
│
├── RAFT_GUIDE.md               # Comprehensive RAFT guide
├── RAFT_QUICK_REFERENCE.md     # Quick reference card
├── RAG_EXPLAINED.md            # 🆕 Complete RAG explanation
├── RAG_QUICKSTART.md           # 🆕 RAG quick start guide
├── README.md                   # This file
│
├── config.yaml                 # Training configuration
├── requirements.txt            # Python dependencies
└── roberto_data.txt            # Source training data
```

---

## 🎓 How RAFT Works

### Traditional Fine-tuning

```
Input:  "Roberto Arce is a"
Output: "Data Scientist and Machine Learning Engineer..."
```

- Learns to complete text
- Risk of hallucination
- No explicit context handling

### RAFT Fine-tuning

```
Input:  Documents: [Doc1, Doc2, Doc3]
        Question: "Where does Roberto work?"
Output: "Based on the documents: Roberto works at Sanofi as a Data Scientist."
```

- Learns to extract from context
- Grounded in documents
- Ignores distractors

### Key Components

1. **Oracle Document** - Contains the actual answer
2. **Distractor Documents** - Related but don't contain the answer (optional)
3. **Question** - What needs to be answered
4. **Training** - Model learns to identify and use the oracle document

---

## 💻 Usage Examples

### Example 1: Generate RAFT Dataset Only

```bash
python raft_data_generator.py
```

Customize in the script:

```python
generator = RAFTDataGenerator(
    source_file="roberto_data.txt",
    output_file="raft_training_data.jsonl",
    chunk_size=300,              # Document chunk size
    num_distractors=3,           # Number of distractors per example
    distractor_probability=0.5   # 50% of examples include distractors
)
```

### Example 2: Fine-tune with Existing Dataset

```bash
python raft_fine_tuning.py
```

### Example 3: Interactive Testing

```python
from test_raft_model import RAFTModelTester

# Load model
tester = RAFTModelTester("./finetuned_roberto_raft_20250103_120000")

# Create query
question = "What programming languages does Roberto know?"
documents = [
    "Roberto's expertise includes Python, JavaScript, Vue.js, and SQL.",
    "He's proficient in ML frameworks like scikit-learn and TensorFlow.",
    "Roberto studied Industrial Engineering."  # Distractor
]

# Generate answer
prompt = tester.create_raft_prompt(question, documents)
answer = tester.generate_answer(prompt)
print(answer)
# Output: "Based on the documents, Roberto knows Python, JavaScript, Vue.js, and SQL..."
```

### Example 4: See RAFT Demo

```bash
python example_raft_usage.py
```

Demonstrates:

- RAFT data format
- Training vs inference
- Complete workflow
- Best practices

---

## ⚙️ Configuration

### Edit `config.yaml` to customize:

```yaml
# Model Settings
model:
  name: "distilgpt2" # distilgpt2, gpt2, gpt2-medium, gpt2-large
  max_length: 512 # Max token length
  fp16: true # Mixed precision training

# Training Settings
training:
  num_epochs: 5 # Number of training epochs
  batch_size: 8 # Batch size per device
  learning_rate: 5e-5 # Learning rate
  gradient_accumulation_steps: 2

# Data Settings
data:
  file_path: "roberto_data.txt"
  train_split: 0.9 # 90% train, 10% validation

# Output Settings
output:
  base_dir: "./finetuned_roberto"
  save_total_limit: 3 # Keep only 3 best checkpoints
```

---

## 📊 RAFT vs Traditional Fine-tuning

| Feature                | Traditional              | RAFT                              |
| ---------------------- | ------------------------ | --------------------------------- |
| **Training Style**     | Text completion          | Question + Context → Answer       |
| **Use Case**           | Creative text generation | Question answering with documents |
| **Hallucination Risk** | Higher                   | Lower (grounded in context)       |
| **Context Handling**   | Implicit                 | Explicit (documents provided)     |
| **RAG Integration**    | Separate step            | Built-in during training          |
| **Best For**           | General text generation  | Factual Q&A, document extraction  |

---

## 📈 Expected Results

### ✅ Signs of Success

- Model extracts information from provided documents
- Refuses to answer when information not available
- Ignores distractor documents
- Low hallucination rate
- Cites relevant context

### ⚠️ Warning Signs

- Always returns same generic answer
- Ignores provided documents completely
- Never refuses to answer
- Makes up information not in documents

---

## 🛠️ Troubleshooting

| Issue                       | Solution                                            |
| --------------------------- | --------------------------------------------------- |
| **Out of memory error**     | Reduce `batch_size` and `max_length` in config.yaml |
| **Model ignores context**   | Increase `num_epochs`, verify data format           |
| **Never refuses to answer** | Increase `distractor_probability` to 0.6-0.8        |
| **Training too slow**       | Use smaller model (`distilgpt2`), enable `fp16`     |
| **Poor answer quality**     | Add more diverse training examples                  |

---

## 🔗 Complete RAG System Included! 🆕

This project now includes a **complete RAG (Retrieval Augmented Generation) system** that integrates seamlessly with your RAFT model!

### What's Included

```python
from rag_system import RAGSystem

# Initialize RAG with RAFT model
rag = RAGSystem(
    source_file="roberto_data.txt",
    raft_model_path="./finetuned_roberto_raft_20250103_120000"
)

# Create vector database (one-time)
rag.create_vector_database()

# Query the system
result = rag.query("Where does Roberto work?")
print(result['answer'])
# "Based on the documents, Roberto works at Sanofi as a Data Scientist..."
```

### How RAG + RAFT Works Together

```
User Question
     ↓
[1. RETRIEVE] Vector DB finds relevant documents (ChromaDB + Embeddings)
     ↓
[2. AUGMENT] Format documents with question in RAFT style
     ↓
[3. GENERATE] RAFT model extracts answer from context
     ↓
Answer with Sources
```

### Why Use RAG + RAFT?

| Feature                   | RAFT Only | RAG Only | RAFT + RAG ✨ |
| ------------------------- | --------- | -------- | ------------- |
| **Dynamic Knowledge**     | ❌        | ✅       | ✅            |
| **Context Extraction**    | ✅        | ⚠️       | ✅            |
| **Reduced Hallucination** | ✅        | ⚠️       | ✅✅          |
| **Source Citations**      | ❌        | ✅       | ✅            |
| **Scales to Large Docs**  | ❌        | ✅       | ✅            |
| **Best Answer Quality**   | ⚠️        | ⚠️       | ✅            |

### Quick Start with RAG

```bash
# Install RAG dependencies
pip install chromadb sentence-transformers

# Run the RAG system
python rag_system.py
```

See **[RAG_QUICKSTART.md](RAG_QUICKSTART.md)** for detailed instructions!

**Benefits:**

- ✅ **Better context utilization** - RAFT trained to use documents
- ✅ **Reduced hallucination** - Grounded in retrieved context
- ✅ **Source citations** - Know where answers come from
- ✅ **Scalable** - Add documents without retraining
- ✅ **Production-ready** - Complete working implementation

---

## 📚 Documentation

### RAFT Documentation

- **📖 [RAFT_GUIDE.md](RAFT_GUIDE.md)** - Comprehensive guide with best practices and advanced techniques
- **⚡ [RAFT_QUICK_REFERENCE.md](RAFT_QUICK_REFERENCE.md)** - Quick reference card with commands
- **💡 [example_raft_usage.py](example_raft_usage.py)** - Interactive demo and tutorial

### RAG Documentation 🆕

- **📖 [RAG_EXPLAINED.md](RAG_EXPLAINED.md)** - Complete RAG explanation with architecture diagrams
- **⚡ [RAG_QUICKSTART.md](RAG_QUICKSTART.md)** - Get RAG running in 5 minutes
- **💻 [rag_system.py](rag_system.py)** - Full RAG implementation code

---

## 🎯 Common Use Cases

1. **Domain-Specific Q&A**

   - Customer support bots
   - Technical documentation assistants
   - Medical/legal information extraction

2. **RAG Enhancement**

   - Improve existing RAG systems
   - Better context utilization
   - Reduced hallucination

3. **Document Analysis**

   - Extract facts from reports
   - Summarize with citations
   - Compare multiple sources

4. **Knowledge Base Queries**
   - Internal company knowledge
   - Product information
   - Research papers

---

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 8GB+ RAM (16GB recommended)
- GPU recommended (but CPU works for small models)

See `requirements.txt` for complete dependencies.

---

## 📝 Training Data Format

RAFT generates training examples in JSONL format:

```json
{
  "instruction": "Use the following documents to answer...\n\nDocument [1]: ...\n\nQuestion: ...\nAnswer:",
  "output": "Based on the documents: [extracted answer]",
  "question": "Where does Roberto work?",
  "oracle_context": "The document containing the answer",
  "num_distractors": 3,
  "has_distractors": true
}
```

---

## 🎓 Best Practices

1. **Start Small**: Use `distilgpt2` for initial experiments
2. **Quality Data**: Clean, well-structured source documents
3. **Mix Training**: 50% with distractors, 50% without
4. **Diverse Questions**: Factual, comparative, descriptive
5. **Validate Often**: Test on unseen questions regularly
6. **Monitor Metrics**: Watch eval_loss during training

---

## 🚀 Next Steps

### For Beginners

1. ✅ **Understand concepts**: `python example_raft_usage.py`
2. ✅ **Train RAFT model**: `python run_raft.py`
3. ✅ **Try RAG system**: `python rag_system.py`
4. ✅ **Read guides**: Check out the documentation

### For Intermediate Users

1. ✅ **Customize data**: Replace `roberto_data.txt` with your documents
2. ✅ **Tune parameters**: Edit `config.yaml` for your needs
3. ✅ **Experiment**: Try different models and chunk sizes
4. ✅ **Evaluate**: Test on your specific use cases

### For Advanced Users

1. ✅ **Optimize retrieval**: Implement hybrid search
2. ✅ **Add re-ranking**: Improve relevance scoring
3. ✅ **Production deployment**: Add API, caching, monitoring
4. ✅ **Fine-tune embeddings**: Domain-specific embedding models
5. ✅ **Scale up**: Distributed vector databases

---

## 📖 Additional Resources

- **Original RAFT Paper**: [arXiv:2403.10131](https://arxiv.org/abs/2403.10131)
- **HuggingFace Transformers**: [Documentation](https://huggingface.co/docs/transformers)
- **RAG Overview**: [Retrieval Augmented Generation](https://huggingface.co/docs/transformers/model_doc/rag)

---

## 🤝 Contributing

Improvements and suggestions welcome! This is a template project for RAFT fine-tuning that can be adapted to any domain.

---

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

## 🙏 Acknowledgments

- RAFT methodology based on the research paper by Anthropic
- Built with HuggingFace Transformers
- Trained on domain-specific professional data

---

**Ready to get started?** Run `python run_raft.py` and watch the magic happen! ✨

For detailed information, check out [RAFT_GUIDE.md](RAFT_GUIDE.md)
