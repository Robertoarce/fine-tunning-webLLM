"""
RAFT Data Generator
Creates training data in RAFT format from source documents.

RAFT (Retrieval Augmented Fine Tuning) creates training examples with:
- Question/Instruction
- Relevant context documents (oracle documents)
- Distractor documents (optional)
- Answer generated from the oracle document
"""

import json
import random
import re
from typing import List, Dict, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAFTDataGenerator:
    """Generate RAFT-formatted training data from source documents"""
    
    def __init__(
        self,
        source_file: str,
        output_file: str = "raft_training_data.jsonl",
        chunk_size: int = 300,
        num_distractors: int = 3,
        distractor_probability: float = 0.5
    ):
        """
        Initialize RAFT data generator
        
        Args:
            source_file: Path to source text file
            output_file: Path to output JSONL file
            chunk_size: Approximate size of text chunks (in words)
            num_distractors: Number of distractor documents to include
            distractor_probability: Probability of including distractors (0.0 to 1.0)
        """
        self.source_file = source_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.num_distractors = num_distractors
        self.distractor_probability = distractor_probability
        self.chunks = []
        
    def load_and_chunk_documents(self) -> List[str]:
        """Load source file and split into chunks"""
        logger.info(f"Loading documents from {self.source_file}")
        
        with open(self.source_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split by sections (using headers as delimiters)
        sections = re.split(r'\n#{1,3}\s+', text)
        
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Split long sections into smaller chunks
            words = section.split()
            if len(words) > self.chunk_size:
                for i in range(0, len(words), self.chunk_size):
                    chunk = ' '.join(words[i:i + self.chunk_size])
                    if chunk:
                        chunks.append(chunk)
            else:
                chunks.append(section)
        
        self.chunks = [c for c in chunks if len(c.split()) > 20]  # Filter out very small chunks
        logger.info(f"Created {len(self.chunks)} document chunks")
        return self.chunks
    
    def generate_qa_pairs(self, chunk: str) -> List[Tuple[str, str]]:
        """
        Generate question-answer pairs from a chunk
        
        Args:
            chunk: Text chunk to generate QA pairs from
            
        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        
        # Generate different types of questions based on content patterns
        
        # 1. Factual questions about skills, tools, technologies
        tech_patterns = re.findall(r'(Python|JavaScript|SQL|Docker|AWS|TensorFlow|PyTorch|scikit-learn|Pandas|NumPy|Vue\.js|Plotly|Tableau|PostgreSQL|MongoDB|Redis|Azure|GCP)', chunk, re.IGNORECASE)
        if tech_patterns:
            unique_techs = list(set(tech_patterns))
            for tech in unique_techs[:2]:  # Limit to 2 per chunk
                qa_pairs.append((
                    f"What experience does Roberto have with {tech}?",
                    chunk
                ))
        
        # 2. Educational background questions
        if any(word in chunk.lower() for word in ['master', 'bachelor', 'degree', 'msc', 'education']):
            qa_pairs.append((
                "What is Roberto's educational background?",
                chunk
            ))
            if 'industrial engineering' in chunk.lower():
                qa_pairs.append((
                    "What did Roberto study for his Bachelor's degree?",
                    chunk
                ))
        
        # 3. Professional experience questions
        if any(word in chunk.lower() for word in ['work', 'experience', 'sanofi', 'data scientist', 'engineer']):
            qa_pairs.append((
                "Where does Roberto work and what does he do?",
                chunk
            ))
        
        # 4. Project-specific questions
        if 'project' in chunk.lower():
            project_names = re.findall(r'###\s+([^\n]+)', chunk)
            for project in project_names:
                qa_pairs.append((
                    f"Tell me about Roberto's {project} project",
                    chunk
                ))
        
        # 5. Certification questions
        if any(word in chunk.lower() for word in ['certification', 'specialization', 'course']):
            qa_pairs.append((
                "What certifications does Roberto have?",
                chunk
            ))
        
        # 6. General competency questions
        if 'skill' in chunk.lower() or 'expertise' in chunk.lower():
            qa_pairs.append((
                "What are Roberto's technical skills?",
                chunk
            ))
        
        # 7. Generic extraction questions
        qa_pairs.append((
            "Provide information about Roberto Arce based on the following context",
            chunk
        ))
        
        return qa_pairs
    
    def create_raft_example(
        self,
        question: str,
        oracle_chunk: str,
        all_chunks: List[str]
    ) -> Dict:
        """
        Create a single RAFT training example
        
        Args:
            question: The question/instruction
            oracle_chunk: The chunk containing the answer (oracle document)
            all_chunks: All available chunks for selecting distractors
            
        Returns:
            Dictionary with RAFT training example
        """
        # Decide whether to include distractors
        include_distractors = random.random() < self.distractor_probability
        
        context_docs = [oracle_chunk]
        
        if include_distractors:
            # Select random distractor documents
            available_distractors = [c for c in all_chunks if c != oracle_chunk]
            num_distractors = min(self.num_distractors, len(available_distractors))
            distractors = random.sample(available_distractors, num_distractors)
            context_docs.extend(distractors)
            
            # Shuffle so oracle isn't always first
            random.shuffle(context_docs)
        
        # Format the context
        formatted_context = "\n\n".join([
            f"Document [{i+1}]:\n{doc}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Create the instruction with context
        instruction = f"""Use the following documents to answer the question. If the answer cannot be found in the documents, say "I cannot answer based on the provided context."

{formatted_context}

Question: {question}
Answer:"""
        
        # The answer is based on the oracle chunk
        answer = f"Based on the documents: {oracle_chunk}"
        
        # Create training example in instruction format
        training_example = {
            "instruction": instruction,
            "output": answer,
            "question": question,
            "oracle_context": oracle_chunk,
            "num_distractors": len(context_docs) - 1,
            "has_distractors": include_distractors
        }
        
        return training_example
    
    def generate_dataset(self) -> List[Dict]:
        """Generate complete RAFT dataset"""
        logger.info("Generating RAFT dataset...")
        
        if not self.chunks:
            self.load_and_chunk_documents()
        
        dataset = []
        
        for chunk in self.chunks:
            qa_pairs = self.generate_qa_pairs(chunk)
            
            for question, answer_chunk in qa_pairs:
                example = self.create_raft_example(
                    question=question,
                    oracle_chunk=answer_chunk,
                    all_chunks=self.chunks
                )
                dataset.append(example)
        
        logger.info(f"Generated {len(dataset)} RAFT training examples")
        return dataset
    
    def save_dataset(self, dataset: List[Dict]):
        """Save dataset to JSONL file"""
        logger.info(f"Saving dataset to {self.output_file}")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(dataset)} examples to {self.output_file}")
        
        # Print statistics
        with_distractors = sum(1 for ex in dataset if ex['has_distractors'])
        without_distractors = len(dataset) - with_distractors
        
        logger.info(f"Examples with distractors: {with_distractors}")
        logger.info(f"Examples without distractors: {without_distractors}")
    
    def generate_and_save(self) -> str:
        """Generate and save complete RAFT dataset"""
        dataset = self.generate_dataset()
        self.save_dataset(dataset)
        return self.output_file


def main():
    """Main function to generate RAFT dataset"""
    generator = RAFTDataGenerator(
        source_file="roberto_data.txt",
        output_file="raft_training_data.jsonl",
        chunk_size=300,
        num_distractors=3,
        distractor_probability=0.5  # 50% of examples will have distractors
    )
    
    output_file = generator.generate_and_save()
    
    # Preview first example
    logger.info("\n" + "="*80)
    logger.info("Preview of first training example:")
    logger.info("="*80)
    
    with open(output_file, 'r') as f:
        first_example = json.loads(f.readline())
        print(f"\nQuestion: {first_example['question']}")
        print(f"\nHas distractors: {first_example['has_distractors']}")
        print(f"Number of distractors: {first_example['num_distractors']}")
        print(f"\nInstruction (first 500 chars):\n{first_example['instruction'][:500]}...")


if __name__ == "__main__":
    main()

