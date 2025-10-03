"""
Simple Example: RAFT Data Generation and Usage
Demonstrates the core concepts of RAFT with a minimal example.
"""

import json


def create_simple_raft_example():
    """
    Create a simple RAFT training example to demonstrate the format
    """

    # Oracle document - contains the answer
    oracle_doc = """
    Roberto Arce is a Data Scientist and Machine Learning Engineer based in France. 
    He currently works at Sanofi (2023-Present), providing data science services to 
    various internal clients with a focus on machine learning solutions and data analysis.
    """

    # Distractor documents - related but don't answer the question
    distractor_1 = """
    Roberto has multiple Master's degrees: MSc Supply Chain, MSc Finance, and MSc Management. 
    He also has a Bachelor's in Industrial Engineering.
    """

    distractor_2 = """
    Roberto's technical skills include Python, JavaScript, Vue.js, and SQL. He's proficient 
    in machine learning frameworks like scikit-learn, TensorFlow, and PyTorch.
    """

    # Question about the oracle document
    question = "Where does Roberto currently work and what is his role?"

    # Format as RAFT training example
    formatted_context = f"""Document [1]:
{distractor_1}

Document [2]:
{oracle_doc}

Document [3]:
{distractor_2}"""

    instruction = f"""Use the following documents to answer the question. If the answer cannot be found in the documents, say "I cannot answer based on the provided context."

{formatted_context}

Question: {question}
Answer:"""

    # Expected answer (based on oracle document)
    answer = "Based on the documents: Roberto currently works at Sanofi as a Data Scientist and Machine Learning Engineer. He has been there since 2023, providing data science services to various internal clients with a focus on machine learning solutions and data analysis."

    # Complete RAFT training example
    raft_example = {
        "instruction": instruction,
        "output": answer,
        "question": question,
        "oracle_context": oracle_doc.strip(),
        "num_distractors": 2,
        "has_distractors": True
    }

    return raft_example


def demonstrate_raft_format():
    """
    Demonstrate what RAFT training data looks like
    """
    print("="*80)
    print("RAFT Training Example Demonstration")
    print("="*80)

    example = create_simple_raft_example()

    print("\n📋 RAFT TRAINING EXAMPLE STRUCTURE:")
    print("-"*80)

    print("\n1️⃣ INSTRUCTION (what the model sees as input):")
    print("-"*80)
    print(example['instruction'][:400] + "...\n")

    print("\n2️⃣ OUTPUT (what the model should generate):")
    print("-"*80)
    print(example['output'])

    print("\n\n📊 METADATA:")
    print("-"*80)
    print(f"Question: {example['question']}")
    print(f"Has Distractors: {example['has_distractors']}")
    print(f"Number of Distractors: {example['num_distractors']}")
    print(f"\nOracle Context (first 150 chars):")
    print(example['oracle_context'][:150] + "...")

    print("\n\n💾 SAVING TO JSON FORMAT:")
    print("-"*80)
    json_output = json.dumps(example, indent=2)
    print(json_output[:400] + "...\n")

    print("\n✅ KEY TAKEAWAYS:")
    print("-"*80)
    print("1. RAFT includes multiple documents (oracle + distractors)")
    print("2. The model learns to identify relevant information")
    print("3. Training teaches the model to extract answers from context")
    print("4. Distractors make the model more robust and reduce hallucination")
    print("5. The format is ideal for Q&A and RAG applications")
    print("="*80)


def show_training_vs_inference():
    """
    Show the difference between training and inference
    """
    print("\n\n" + "="*80)
    print("TRAINING vs INFERENCE")
    print("="*80)

    print("\n🏋️ DURING TRAINING:")
    print("-"*80)
    print("Input: Instruction + Multiple Documents + Question")
    print("Output: Expected Answer based on Oracle Document")
    print("Goal: Learn to extract relevant info and ignore distractors")

    print("\n\n🔮 DURING INFERENCE (using the fine-tuned model):")
    print("-"*80)
    print("Input: Instruction + Multiple Documents + Question")
    print("Output: Generated Answer (model predicts this)")
    print("Benefit: Model knows how to use context and avoid hallucination")

    print("\n\n📈 EXAMPLE PROGRESSION:")
    print("-"*80)

    # Before RAFT training
    print("\n❌ BEFORE RAFT Training (base model):")
    print("Q: Where does Roberto work?")
    print("Docs: [Contains Sanofi info]")
    print("A: Roberto works as an engineer... [hallucinates or ignores docs]")

    # After RAFT training
    print("\n✅ AFTER RAFT Training:")
    print("Q: Where does Roberto work?")
    print("Docs: [Contains Sanofi info]")
    print("A: Based on the documents, Roberto works at Sanofi as a Data Scientist.")

    print("\n" + "="*80)


def show_complete_workflow():
    """
    Show the complete RAFT workflow
    """
    print("\n\n" + "="*80)
    print("COMPLETE RAFT WORKFLOW")
    print("="*80)

    workflow = """
    Step 1: PREPARE SOURCE DATA
    ├── Load your domain-specific text (e.g., roberto_data.txt)
    └── Ensure it's well-structured with sections/headers
    
    Step 2: GENERATE RAFT DATASET
    ├── Run: python raft_data_generator.py
    ├── Chunks documents into semantic sections
    ├── Generates questions for each chunk
    ├── Creates training examples with oracle + distractors
    └── Output: raft_training_data.jsonl
    
    Step 3: FINE-TUNE MODEL
    ├── Run: python raft_fine_tuning.py
    ├── Loads RAFT dataset
    ├── Formats for causal language modeling
    ├── Trains model to extract answers from context
    └── Output: finetuned_roberto_raft_[timestamp]/
    
    Step 4: TEST MODEL
    ├── Run: python test_raft_model.py
    ├── Load fine-tuned model
    ├── Test with Q&A examples
    └── Verify it uses context correctly
    
    Step 5: DEPLOY & USE
    ├── Integrate with RAG system (optional)
    ├── Format user queries in RAFT style
    ├── Get grounded, context-based answers
    └── Enjoy reduced hallucination! 🎉
    """

    print(workflow)
    print("="*80)


def main():
    """
    Run all demonstrations
    """
    print("\n" + "🚀 RAFT (Retrieval Augmented Fine Tuning) - Interactive Demo\n")

    # Show what RAFT training data looks like
    demonstrate_raft_format()

    # Show training vs inference
    show_training_vs_inference()

    # Show complete workflow
    show_complete_workflow()

    print("\n\n💡 NEXT STEPS:")
    print("-"*80)
    print("1. Run: python raft_data_generator.py")
    print("   → Generates RAFT dataset from your data")
    print()
    print("2. Run: python run_raft.py")
    print("   → Complete pipeline (generates data + trains model)")
    print()
    print("3. Run: python test_raft_model.py")
    print("   → Test your fine-tuned RAFT model")
    print()
    print("4. Read: RAFT_GUIDE.md")
    print("   → Complete guide with best practices")
    print("="*80)


if __name__ == "__main__":
    main()
