"""
Test RAFT Fine-tuned Model
Interactive script to test the RAFT fine-tuned model with custom queries.
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAFTModelTester:
    """Test RAFT fine-tuned models"""
    
    def __init__(self, model_path: str):
        """Initialize with model path"""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model"""
        logger.info(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        logger.info("Model loaded successfully!")
    
    def create_raft_prompt(self, question: str, documents: list) -> str:
        """
        Create a RAFT-style prompt with documents and question
        
        Args:
            question: The question to answer
            documents: List of document strings to use as context
            
        Returns:
            Formatted prompt string
        """
        formatted_docs = "\n\n".join([
            f"Document [{i+1}]:\n{doc}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""Use the following documents to answer the question. If the answer cannot be found in the documents, say "I cannot answer based on the provided context."

{formatted_docs}

Question: {question}
Answer:"""
        
        return prompt
    
    def generate_answer(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """Generate answer for the given prompt"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_new_tokens,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=top_p,
                repetition_penalty=1.1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the answer (everything after "Answer:")
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        return answer
    
    def test_with_examples(self):
        """Test model with pre-defined examples"""
        
        test_cases = [
            {
                "question": "Where does Roberto work and what is his role?",
                "documents": [
                    "Roberto Arce is a Data Scientist and Machine Learning Engineer based in France. Currently, he works at Sanofi (2023-Present), providing data science services to various internal clients with a focus on machine learning solutions and data analysis.",
                    "Roberto has expertise in Python, JavaScript, and SQL. He's proficient in machine learning frameworks."
                ]
            },
            {
                "question": "What programming languages does Roberto know?",
                "documents": [
                    "Roberto's programming expertise includes Python, JavaScript, Vue.js, and SQL. He's proficient in machine learning frameworks like scikit-learn, TensorFlow, PyTorch, Pandas, and NumPy.",
                    "Roberto studied Industrial Engineering for his Bachelor's degree, which provided solid thinking foundations."
                ]
            },
            {
                "question": "What is Roberto's educational background?",
                "documents": [
                    "Roberto has multiple Master's degrees: MSc Supply Chain (Global Perspective Framework), MSc Finance (Quantitative financial management), and MSc Management (Business understanding). He also has a Bachelor's in Industrial Engineering.",
                    "Roberto works with tools like Docker, Git, Jupyter, Tableau, and Power BI."
                ]
            },
            {
                "question": "What machine learning certifications does Roberto have?",
                "documents": [
                    "Roberto has completed the Machine Learning Specialization by Andrew Ng from Stanford, which covers supervised learning, unsupervised learning, and best practices for ML development.",
                    "He also completed the Deep Learning Specialization by Andrew Ng, covering neural networks, CNNs, RNNs, GANs, and transformer architectures.",
                    "Roberto works at Sanofi as a Data Scientist since 2023."
                ]
            }
        ]
        
        logger.info("\n" + "="*80)
        logger.info("Testing RAFT Model with Examples")
        logger.info("="*80)
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n--- Test Case {i} ---")
            logger.info(f"Question: {test_case['question']}")
            
            prompt = self.create_raft_prompt(
                question=test_case['question'],
                documents=test_case['documents']
            )
            
            answer = self.generate_answer(prompt)
            
            logger.info(f"Answer: {answer}\n")
    
    def interactive_mode(self):
        """Interactive mode for custom questions"""
        logger.info("\n" + "="*80)
        logger.info("RAFT Interactive Mode")
        logger.info("="*80)
        logger.info("Enter your question and documents to get answers.")
        logger.info("Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                if question.lower() == 'quit':
                    break
                
                if not question:
                    continue
                
                print("\nEnter documents (one per line, empty line to finish):")
                documents = []
                while True:
                    doc = input().strip()
                    if not doc:
                        break
                    documents.append(doc)
                
                if not documents:
                    print("No documents provided. Please provide at least one document.\n")
                    continue
                
                prompt = self.create_raft_prompt(question, documents)
                answer = self.generate_answer(prompt)
                
                print(f"\nAnswer: {answer}\n")
                print("-"*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def find_latest_model():
    """Find the most recently created RAFT model"""
    model_dirs = list(Path(".").glob("finetuned_roberto_raft_*"))
    
    if not model_dirs:
        return None
    
    # Sort by modification time
    latest = max(model_dirs, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    """Main testing function"""
    
    # Check for command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Try to find latest model
        model_path = find_latest_model()
        if not model_path:
            logger.error("No RAFT model found. Please train a model first or specify path.")
            logger.error("Usage: python test_raft_model.py [model_path]")
            sys.exit(1)
        logger.info(f"Using latest model: {model_path}")
    
    # Initialize tester
    tester = RAFTModelTester(model_path)
    
    # Run predefined tests
    tester.test_with_examples()
    
    # Interactive mode
    response = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        tester.interactive_mode()
    
    logger.info("\nTesting complete!")


if __name__ == "__main__":
    main()

