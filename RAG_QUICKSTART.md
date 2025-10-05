# RAG System - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install chromadb sentence-transformers
```

Or install everything:

```bash
pip install -r requirements.txt
```

### Step 2: Run RAG System

```bash
python rag_system.py
```

This will:

1. ✅ Load your documents (`roberto_data.txt`)
2. ✅ Create vector database (ChromaDB)
3. ✅ Find and load your RAFT model (if available)
4. ✅ Run example queries
5. ✅ Launch interactive mode

---

## 📝 What You'll See

### Example Output:

```
================================================================================
RAG (Retrieval Augmented Generation) System
================================================================================

✓ Found RAFT model: ./finetuned_roberto_raft_20250103_120000
Use this RAFT model? (y/n): y

Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Initializing vector database...
Loading RAFT model from ./finetuned_roberto_raft_20250103_120000
RAFT model loaded successfully!

Loading documents from roberto_data.txt
Created 45 document chunks

Creating vector database...
Generating embeddings...
✓ Vector database created with 45 documents

================================================================================
RAG System Ready! 🚀
================================================================================

Running example queries...

================================================================================
Query: Where does Roberto work and what is his role?
================================================================================

[1/3] Retrieving relevant documents...
✓ Retrieved 3 documents
  1. Professional Experience (distance: 0.3421)
  2. About Me (distance: 0.4156)
  3. Key Projects (distance: 0.5234)

[2/3] Creating RAFT prompt...
✓ Prompt created (1234 chars)

[3/3] Generating answer with RAFT model...

================================================================================
Answer:
================================================================================
Based on the documents, Roberto works at Sanofi as a Data Scientist, a position
he has held since 2023. In this role, he provides data science services to
various internal clients with a focus on machine learning solutions and data
analysis.
================================================================================

💬 Your question: _
```

---

## 💡 Example Queries

Try asking:

### About Work

- "Where does Roberto work?"
- "What does Roberto do at his current job?"
- "Tell me about Roberto's professional experience"

### About Skills

- "What programming languages does Roberto know?"
- "What machine learning frameworks is Roberto proficient in?"
- "What tools does Roberto use?"

### About Education

- "What is Roberto's educational background?"
- "What degrees does Roberto have?"
- "Tell me about Roberto's certifications"

### About Projects

- "What projects has Roberto worked on?"
- "Describe Roberto's ML pipeline project"
- "What technologies did Roberto use in his projects?"

---

## 🔧 How It Works

### 1. You Ask a Question

```
💬 Your question: What programming languages does Roberto know?
```

### 2. System Retrieves Relevant Documents

```
[Retrieval] Searching vector database...
Found 3 relevant documents:
- Technical Skills (similarity: 0.89)
- About Me (similarity: 0.76)
- Key Projects (similarity: 0.65)
```

### 3. RAFT Model Generates Answer

```
[Generation] Using RAFT model with context...

Answer: Based on the documents, Roberto's programming expertise
includes Python, JavaScript, Vue.js, and SQL. He's proficient in
machine learning frameworks like scikit-learn, TensorFlow, PyTorch,
Pandas, and NumPy.
```

### 4. Sources Are Shown

```
Sources:
[1] Technical Skills
    My programming expertise includes Python, JavaScript, Vue.js...
[2] About Me
    Hi, I'm Roberto Arce, a Data Scientist...
```

---

## ⚙️ Customization

### Change Number of Retrieved Documents

Edit in `rag_system.py`:

```python
rag.query(question, top_k=5)  # Retrieve 5 documents instead of 3
```

### Change Embedding Model

```python
rag = RAGSystem(
    embedding_model="sentence-transformers/all-mpnet-base-v2"  # Better quality
)
```

Available models:

- `all-MiniLM-L6-v2` - Fast, good quality (default)
- `all-mpnet-base-v2` - Better quality, slower
- `all-MiniLM-L12-v2` - Balanced

### Adjust Chunk Size

```python
rag = RAGSystem(
    chunk_size=200  # Smaller chunks (more precise)
    # or
    chunk_size=500  # Larger chunks (more context)
)
```

### Use Different Source Files

```python
rag = RAGSystem(
    source_file="my_documents.txt"
)
```

---

## 🎯 Use Programmatically

### Basic Usage

```python
from rag_system import RAGSystem

# Initialize
rag = RAGSystem(
    source_file="roberto_data.txt",
    raft_model_path="./finetuned_roberto_raft_20250103_120000"
)

# Create vector database
rag.create_vector_database()

# Query
result = rag.query("Where does Roberto work?", verbose=False)

print(result['answer'])
print(f"Sources: {len(result['sources'])}")
```

### Advanced Usage

```python
# Retrieve documents only (no generation)
docs = rag.retrieve("machine learning", top_k=5)
for doc in docs:
    print(f"- {doc['text'][:100]}...")

# Custom prompt and generation
prompt = rag.create_raft_prompt("Your question?", docs)
answer = rag.generate_answer(prompt, temperature=0.5)
print(answer)
```

### Batch Processing

```python
questions = [
    "Where does Roberto work?",
    "What skills does Roberto have?",
    "What is Roberto's education?"
]

results = []
for q in questions:
    result = rag.query(q, verbose=False)
    results.append(result)

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 🔍 Understanding the Output

### Answer Fields

```python
result = {
    'question': 'Where does Roberto work?',
    'answer': 'Based on the documents, Roberto works at Sanofi...',
    'sources': [
        {
            'id': 'chunk_5',
            'text': 'Roberto works at Sanofi...',
            'metadata': {'title': 'Professional Experience', 'section': 2},
            'distance': 0.3421  # Lower = more similar
        },
        # ... more sources
    ],
    'prompt': 'Use the following documents...'  # Full RAFT prompt used
}
```

### Distance/Similarity Score

- **0.0 - 0.3**: Highly relevant ✅
- **0.3 - 0.6**: Moderately relevant ⚠️
- **0.6 - 1.0**: Less relevant ❌

---

## 🚨 Troubleshooting

### Error: "No RAFT model found"

**Solution:**

```bash
# Train a RAFT model first
python run_raft.py

# Or specify path manually
python -c "from rag_system import RAGSystem; rag = RAGSystem(raft_model_path='path/to/model')"
```

### Error: "chromadb not installed"

**Solution:**

```bash
pip install chromadb
```

### Error: "sentence-transformers not installed"

**Solution:**

```bash
pip install sentence-transformers
```

### Slow Embedding Generation

**Solutions:**

1. Use GPU if available (automatic)
2. Use smaller embedding model:
   ```python
   RAGSystem(embedding_model="all-MiniLM-L6-v2")  # Faster
   ```
3. Reduce chunk size:
   ```python
   RAGSystem(chunk_size=200)
   ```

### Poor Answer Quality

**Solutions:**

1. Train RAFT model if not done:
   ```bash
   python run_raft.py
   ```
2. Increase retrieved documents:
   ```python
   rag.query(question, top_k=5)
   ```
3. Improve source documents (clean, structure)

---

## 📊 Performance Tips

### For Speed

- ✅ Use smaller embedding model (`all-MiniLM-L6-v2`)
- ✅ Reduce `top_k` to 3
- ✅ Use smaller RAFT model (`distilgpt2`)
- ✅ Enable GPU acceleration

### For Quality

- ✅ Use better embedding model (`all-mpnet-base-v2`)
- ✅ Increase `top_k` to 5
- ✅ Use larger RAFT model (`gpt2-medium`)
- ✅ Fine-tune RAFT with more examples

### For Memory

- ✅ Use smaller models
- ✅ Reduce chunk size
- ✅ Process in batches
- ✅ Clear cache regularly

---

## 🎓 Next Steps

### 1. Customize for Your Domain

- Replace `roberto_data.txt` with your documents
- Adjust chunk size for your content
- Fine-tune RAFT on your domain

### 2. Improve Retrieval

- Experiment with embedding models
- Try hybrid search (vector + keyword)
- Implement re-ranking

### 3. Enhance Generation

- Fine-tune RAFT with more examples
- Adjust generation parameters
- Add citation formatting

### 4. Production Deployment

- Add API layer (FastAPI/Flask)
- Implement caching
- Monitor performance
- Add user feedback loop

---

## 📚 Learn More

- **[RAG_EXPLAINED.md](RAG_EXPLAINED.md)** - Deep dive into RAG concepts
- **[RAFT_GUIDE.md](RAFT_GUIDE.md)** - Complete RAFT guide
- **[rag_system.py](rag_system.py)** - Full implementation code

---

## 💬 Common Questions

**Q: Do I need a RAFT model?**
A: No, but it significantly improves answer quality. Without RAFT, you need a base model for generation.

**Q: Can I use my own documents?**
A: Yes! Just replace `roberto_data.txt` with your file or specify a different path.

**Q: How much memory do I need?**
A: Minimum 4GB RAM. 8GB+ recommended. GPU optional but faster.

**Q: Can I use with other LLMs?**
A: Yes! Modify `generate_answer()` to use OpenAI, Claude, or other APIs.

**Q: Is this production-ready?**
A: It's a solid foundation. Add error handling, caching, and monitoring for production.

---

**Ready to start?** Run `python rag_system.py` and ask your first question! 🚀
