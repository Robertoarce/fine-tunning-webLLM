# RAFT (Retrieval Augmented Fine Tuning) - Complete Guide

## Overview

RAFT is a fine-tuning technique that trains language models to answer questions based on provided context documents. It's particularly effective for domain-specific question answering and reduces hallucination by grounding responses in actual documents.

## Key Concepts

### 1. Oracle Documents
- **Definition**: The document(s) that contain the actual answer to the question
- **Purpose**: The model learns to extract relevant information from these documents
- **Example**: If asked "Where does Roberto work?", the oracle doc contains "Roberto works at Sanofi"

### 2. Distractor Documents  
- **Definition**: Documents that are topically related but don't contain the answer
- **Purpose**: Train the model to identify and ignore irrelevant information
- **Example**: Including a document about Roberto's education when asking about his current job
- **Benefit**: Makes the model more robust and reduces false answers

### 3. Training Format
Each RAFT training example includes:
```
Instruction: Use the following documents to answer the question.

Document [1]: [Oracle or Distractor]
Document [2]: [Oracle or Distractor]
...

Question: [Your question]
Answer: [Expected answer based on oracle document]
```

## How RAFT Differs from Traditional Fine-tuning

| Aspect | Traditional Fine-tuning | RAFT |
|--------|------------------------|------|
| **Input Format** | Text completion | Question + Multiple Documents |
| **Learning Objective** | Generate fluent text | Extract answers from context |
| **Context Handling** | Implicit | Explicit (documents provided) |
| **Robustness** | May hallucinate | Learns to distinguish relevant/irrelevant info |
| **Best Use Case** | Creative writing, text generation | Q&A, RAG systems, fact extraction |

## Implementation Details

### Step 1: Data Generation

The `raft_data_generator.py` script:

1. **Chunks the source document** into semantic sections
   - Uses headers as natural boundaries
   - Creates ~300-word chunks (configurable)
   - Filters out very small chunks

2. **Generates questions** for each chunk
   - Factual questions about skills/technologies
   - Educational background queries
   - Professional experience questions
   - Project-specific questions
   - Certification inquiries

3. **Creates training examples** with:
   - Question
   - Oracle document (contains answer)
   - 0-3 distractor documents (configurable)
   - 50% probability of including distractors (configurable)

### Step 2: Fine-tuning

The `raft_fine_tuning.py` script:

1. **Loads RAFT data** from JSONL format
2. **Formats examples** for causal language modeling
3. **Tokenizes** with proper padding and truncation
4. **Fine-tunes** using HuggingFace Trainer
5. **Evaluates** on validation set
6. **Saves** the best performing model

### Step 3: Testing

The `test_raft_model.py` script:

1. **Loads the fine-tuned model**
2. **Formats queries** in RAFT style
3. **Generates answers** with temperature sampling
4. **Extracts** just the answer portion

## Usage Examples

### Example 1: Basic RAFT Query

```python
from test_raft_model import RAFTModelTester

tester = RAFTModelTester("./finetuned_roberto_raft_20250101_120000")

question = "What programming languages does Roberto know?"
documents = [
    "Roberto's programming expertise includes Python, JavaScript, Vue.js, and SQL.",
    "Roberto studied Industrial Engineering for his Bachelor's degree."  # Distractor
]

prompt = tester.create_raft_prompt(question, documents)
answer = tester.generate_answer(prompt)
# Output: "Based on the documents, Roberto knows Python, JavaScript, Vue.js, and SQL."
```

### Example 2: Multiple Relevant Documents

```python
question = "Describe Roberto's work experience"
documents = [
    "Roberto works at Sanofi as a Data Scientist since 2023.",
    "He provides data science services with focus on machine learning solutions.",
    "Roberto has expertise in Python and machine learning frameworks."
]

answer = tester.generate_answer(tester.create_raft_prompt(question, documents))
```

### Example 3: No Relevant Document

```python
question = "What is Roberto's favorite color?"
documents = [
    "Roberto works at Sanofi as a Data Scientist.",
    "He has expertise in Python and machine learning."
]

answer = tester.generate_answer(tester.create_raft_prompt(question, documents))
# Output: "I cannot answer based on the provided context."
```

## Configuration Parameters

### Data Generation Parameters

```python
RAFTDataGenerator(
    source_file="roberto_data.txt",         # Your source document
    output_file="raft_training_data.jsonl", # Output JSONL file
    chunk_size=300,                          # Words per chunk (50-500 recommended)
    num_distractors=3,                       # Distractors per example (0-5)
    distractor_probability=0.5               # 0.0 = no distractors, 1.0 = always include
)
```

**Recommendations:**
- **chunk_size**: 200-400 words for balanced context
- **num_distractors**: 2-4 for good robustness without overwhelming the model
- **distractor_probability**: 0.4-0.6 for mixed training (some with, some without)

### Training Parameters (config.yaml)

```yaml
model:
  name: "distilgpt2"  # Start with smaller model
  max_length: 512      # Must fit question + documents + answer
  fp16: true          # Enable for faster training

training:
  num_epochs: 3-5     # RAFT converges relatively quickly
  batch_size: 4-8     # Depends on GPU memory
  learning_rate: 5e-5 # Standard for GPT models
```

## Best Practices

### 1. Document Quality
- ✅ Use clean, well-structured source documents
- ✅ Include section headers for better chunking
- ✅ Remove redundant information
- ❌ Don't use documents with lots of noise/formatting issues

### 2. Question Generation
- ✅ Generate diverse question types (factual, comparative, descriptive)
- ✅ Ensure questions are answerable from the oracle document
- ✅ Mix simple and complex questions
- ❌ Don't make questions too vague or ambiguous

### 3. Distractor Selection
- ✅ Use topically related but non-answering documents
- ✅ Include varying numbers of distractors (0-4)
- ✅ Shuffle document order randomly
- ❌ Don't use completely unrelated distractors (too easy)
- ❌ Don't use documents that also contain the answer

### 4. Training Strategy
- ✅ Start with smaller models (distilgpt2, gpt2)
- ✅ Use validation set to prevent overfitting
- ✅ Monitor evaluation loss closely
- ✅ Test on unseen questions regularly
- ❌ Don't overtrain (3-5 epochs usually sufficient)

## Evaluation Metrics

### During Training
- **Eval Loss**: Should steadily decrease
- **Perplexity**: Lower is better (measures prediction confidence)

### Post Training
- **Answer Accuracy**: Does it answer correctly from oracle docs?
- **Distractor Resistance**: Does it ignore irrelevant documents?
- **Refusal Rate**: Does it say "I don't know" when appropriate?
- **Hallucination Rate**: Does it make up information not in documents?

## Common Issues & Solutions

### Issue 1: Model Always Returns Same Answer
**Cause**: Overfitting or insufficient diversity in training data
**Solution**: 
- Add more diverse questions
- Increase distractor_probability
- Reduce num_epochs

### Issue 2: Model Ignores Context Documents
**Cause**: Model hasn't learned to use context
**Solution**:
- Ensure training examples are properly formatted
- Increase training epochs
- Check that documents are actually included in prompts

### Issue 3: Model Doesn't Refuse to Answer
**Cause**: Not enough examples where answer isn't present
**Solution**:
- Increase distractor_probability to 0.6-0.8
- Add explicit training examples with no answer
- Fine-tune the refusal prompt

### Issue 4: Out of Memory During Training
**Cause**: max_length too large or batch_size too high
**Solution**:
- Reduce max_length to 256-384
- Reduce batch_size to 2-4
- Enable gradient_accumulation_steps
- Use fp16 or int8 quantization

## Integration with RAG Systems

RAFT models work exceptionally well in RAG (Retrieval Augmented Generation) pipelines:

1. **Retrieval**: Use vector database to find relevant documents
2. **Formatting**: Format documents in RAFT style
3. **Generation**: Pass to RAFT fine-tuned model
4. **Answer**: Model extracts answer from retrieved documents

Example RAG integration:
```python
# 1. Retrieve documents
retrieved_docs = vector_db.search(user_query, top_k=5)

# 2. Format in RAFT style
prompt = raft_model.create_raft_prompt(
    question=user_query,
    documents=retrieved_docs
)

# 3. Generate answer
answer = raft_model.generate_answer(prompt)
```

## Advanced Techniques

### 1. Multi-hop Reasoning
Train with questions requiring multiple documents:
```
Question: Compare Roberto's education and work experience
Documents: [Education doc], [Work doc], [Skills doc]
```

### 2. Citation Generation
Modify output format to include citations:
```
Answer: Roberto works at Sanofi [Doc 1]. He has a Master's in Supply Chain [Doc 2].
```

### 3. Confidence Scoring
Add confidence indicators:
```
Answer: [HIGH CONFIDENCE] Roberto works at Sanofi based on Document 1.
```

### 4. Chain of Thought
Include reasoning in answers:
```
Answer: Looking at Document 1, I see Roberto works at Sanofi. 
Document 2 mentions his role as Data Scientist. 
Therefore, Roberto works at Sanofi as a Data Scientist.
```

## Resources

- **Original RAFT Paper**: [Retrieval Augmented Fine Tuning](https://arxiv.org/abs/2403.10131)
- **HuggingFace Transformers**: [Documentation](https://huggingface.co/docs/transformers)
- **RAG Overview**: [Retrieval Augmented Generation](https://huggingface.co/docs/transformers/model_doc/rag)

## Conclusion

RAFT is a powerful technique for creating domain-specific Q&A models that:
- Ground answers in provided context
- Resist distractors and irrelevant information  
- Reduce hallucination
- Work seamlessly with RAG systems

Start with the basic implementation provided, experiment with different configurations, and iterate based on your specific use case!

