# RAFT Quick Reference Card

## 🚀 Quick Start Commands

```bash
# Complete RAFT pipeline (recommended for first-time users)
python run_raft.py

# Or run steps individually:
python raft_data_generator.py    # Generate dataset
python raft_fine_tuning.py       # Train model
python test_raft_model.py        # Test model

# See example output
python example_raft_usage.py
```

## 📋 File Structure

```
├── raft_data_generator.py    # Converts text → RAFT training data
├── raft_fine_tuning.py       # Fine-tunes model with RAFT
├── run_raft.py               # Complete pipeline
├── test_raft_model.py        # Test trained models
├── example_raft_usage.py     # Demo/tutorial script
├── RAFT_GUIDE.md            # Comprehensive guide
├── config.yaml              # Training configuration
└── roberto_data.txt         # Source data
```

## 🎯 What is RAFT?

**RAFT** = Training models to answer questions using provided documents

**Key Components:**
- **Oracle Document**: Contains the answer ✅
- **Distractor Documents**: Don't contain the answer ⚠️
- **Question**: What to answer ❓
- **Training**: Learn to extract from oracle, ignore distractors 🎓

## 📊 Training Data Format

```json
{
  "instruction": "Use the following documents...\n\nDocument [1]: ...\n\nQuestion: ...\nAnswer:",
  "output": "Based on the documents: [answer]",
  "oracle_context": "The document with the answer",
  "num_distractors": 3,
  "has_distractors": true
}
```

## ⚙️ Key Configuration Parameters

### Data Generation
```python
chunk_size = 300           # Document chunk size (words)
num_distractors = 3        # Distractors per example (0-5)
distractor_probability = 0.5  # How often to include distractors
```

### Training (config.yaml)
```yaml
model:
  name: "distilgpt2"      # Model to fine-tune
  max_length: 512         # Max token length
  
training:
  num_epochs: 5           # Training epochs
  batch_size: 8           # Batch size
  learning_rate: 5e-5     # Learning rate
```

## 💡 Common Use Cases

| Use Case | Why RAFT? |
|----------|-----------|
| **Domain Q&A** | Grounds answers in your documents |
| **RAG Systems** | Better at using retrieved context |
| **Fact Extraction** | Learns to find relevant info |
| **Reducing Hallucination** | Must use provided context |

## 🔍 Testing Your Model

### Programmatic Testing
```python
from test_raft_model import RAFTModelTester

tester = RAFTModelTester("./finetuned_roberto_raft_[timestamp]")

prompt = tester.create_raft_prompt(
    question="Where does Roberto work?",
    documents=["Roberto works at Sanofi...", "He has a PhD..."]
)

answer = tester.generate_answer(prompt)
```

### Interactive Testing
```bash
python test_raft_model.py
# Follow prompts to enter questions and documents
```

## 📈 Expected Results

### ✅ Good Signs
- Model cites information from documents
- Refuses to answer when info not in docs
- Ignores distractor documents
- Low hallucination rate

### ❌ Warning Signs
- Always returns same answer
- Ignores provided documents
- Never refuses to answer
- Makes up information

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size` and `max_length` |
| Model ignores context | Increase `num_epochs`, check formatting |
| Never refuses to answer | Increase `distractor_probability` |
| Poor answer quality | More diverse training data needed |
| Training too slow | Use smaller model (`distilgpt2`) |

## 📚 Key Differences

### RAFT vs Traditional Fine-tuning
```
Traditional:  "Roberto Arce is a..." → "Data Scientist..."
RAFT:        "Docs: [...] Q: Who is Roberto?" → "Based on docs: ..."
```

### RAFT vs RAG
```
RAG:   Retrieve docs → Generate with base model
RAFT:  Retrieve docs → Generate with RAFT-tuned model ✨
```

## 🎓 Best Practices

1. **Start Small**: Use `distilgpt2` for testing
2. **Quality over Quantity**: 100 good examples > 1000 poor ones
3. **Mix Training**: 50% with distractors, 50% without
4. **Test Often**: Validate on unseen questions
5. **Monitor Loss**: Stop if eval_loss stops improving

## 🔗 Integration Example (RAG)

```python
# 1. Retrieve from vector DB
docs = vector_db.search(query, top_k=5)

# 2. Format for RAFT
prompt = create_raft_prompt(query, docs)

# 3. Generate with RAFT model
answer = raft_model.generate(prompt)
```

## 📝 Quick Commands Reference

```bash
# Generate dataset only
python raft_data_generator.py

# Train model only (requires existing dataset)
python raft_fine_tuning.py

# Test specific model
python test_raft_model.py ./finetuned_roberto_raft_20250101_120000

# See example/demo
python example_raft_usage.py

# View comprehensive guide
cat RAFT_GUIDE.md
```

## 🎯 Success Metrics

- **Answer Accuracy**: >80% on validation set
- **Refusal Rate**: 10-20% (when appropriate)
- **Hallucination**: <5%
- **Context Usage**: >90% answers cite documents

## 🌟 Next Steps

1. ✅ Run `python example_raft_usage.py` to understand format
2. ✅ Generate your RAFT dataset
3. ✅ Fine-tune with your data
4. ✅ Test and iterate
5. ✅ Deploy in production!

---

**Need more details?** → Read `RAFT_GUIDE.md`  
**Questions?** → Check troubleshooting section above

