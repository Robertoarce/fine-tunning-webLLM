"""
Complete RAG (Retrieval Augmented Generation) System
with RAFT Model Integration

This script implements a full RAG pipeline:
1. Document ingestion and chunking
2. Vector database creation (ChromaDB)
3. Semantic search retrieval
4. RAFT model integration for generation
5. Interactive query interface
"""

import os
import re
import torch
import logging
import weave
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Vector database and embeddings
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Install with: pip install chromadb")

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

# Transformers for RAFT model
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Weave once per process (safe to call at import time)
weave.init('roberto_arce_/RAFT')


class RAGSystem:
    """Complete RAG system with vector database and RAFT model"""

    def __init__(
        self,
        source_file: str = "roberto_data.txt",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        raft_model_path: Optional[str] = None,
        chunk_size: int = 300,
        collection_name: str = "roberto_knowledge"
    ):
        """
        Initialize RAG system

        Args:
            source_file: Path to source documents
            embedding_model: Sentence transformer model for embeddings
            raft_model_path: Path to fine-tuned RAFT model (optional)
            chunk_size: Size of document chunks (in words)
            collection_name: Name for ChromaDB collection
        """
        self.source_file = source_file
        self.chunk_size = chunk_size
        self.collection_name = collection_name

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers")

        # Initialize vector database
        logger.info("Initializing vector database...")
        if CHROMADB_AVAILABLE:
            self.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            self.collection = None
        else:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb")

        # Initialize RAFT model (optional)
        self.raft_model = None
        self.raft_tokenizer = None
        if raft_model_path:
            self.load_raft_model(raft_model_path)

        self.documents = []
        self.chunks = []

    @weave.op()
    def load_raft_model(self, model_path: str):
        """Load RAFT fine-tuned model"""
        # Config option (read from env for simplicity in this module)
        allow_cpu_env = os.environ.get("ALLOW_CPU", "false").lower() in ("1", "true", "yes")
        if not torch.cuda.is_available() and not allow_cpu_env:
            raise RuntimeError("CUDA GPU not available and ALLOW_CPU env var not set. Aborting.")
        logger.info(f"Loading RAFT model from {model_path}")

        self.raft_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if torch.cuda.is_available():
            self.raft_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": 0}
            )
        else:
            self.raft_model = AutoModelForCausalLM.from_pretrained(
                model_path
            )

        logger.info("RAFT model loaded successfully!")

    @weave.op()
    def load_and_chunk_documents(self) -> List[Dict[str, str]]:
        """
        Load source documents and split into chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        logger.info(f"Loading documents from {self.source_file}")

        with open(self.source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by sections (using headers)
        sections = re.split(r'\n#{1,3}\s+', content)

        chunks = []
        chunk_id = 0

        for section_idx, section in enumerate(sections):
            section = section.strip()
            if not section or len(section.split()) < 20:
                continue

            # Extract section title (first line)
            lines = section.split('\n', 1)
            title = lines[0].strip() if lines else f"Section {section_idx}"
            content = lines[1] if len(lines) > 1 else section

            # Split large sections into smaller chunks
            words = content.split()
            if len(words) > self.chunk_size:
                for i in range(0, len(words), self.chunk_size):
                    chunk_text = ' '.join(words[i:i + self.chunk_size])
                    if chunk_text:
                        chunks.append({
                            'id': f"chunk_{chunk_id}",
                            'text': chunk_text,
                            'title': title,
                            'section': section_idx,
                            'chunk_index': i // self.chunk_size
                        })
                        chunk_id += 1
            else:
                chunks.append({
                    'id': f"chunk_{chunk_id}",
                    'text': content,
                    'title': title,
                    'section': section_idx,
                    'chunk_index': 0
                })
                chunk_id += 1

        self.chunks = chunks
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks

    @weave.op()
    def create_vector_database(self):
        """Create vector database from document chunks"""
        if not self.chunks:
            self.load_and_chunk_documents()

        logger.info("Creating vector database...")

        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass

        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Roberto Arce professional knowledge base"}
        )

        # Generate embeddings for all chunks
        logger.info("Generating embeddings...")
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Add to collection
        logger.info("Adding to vector database...")
        self.collection.add(
            ids=[chunk['id'] for chunk in self.chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{
                'title': chunk['title'],
                'section': chunk['section'],
                'chunk_index': chunk['chunk_index']
            } for chunk in self.chunks]
        )

        logger.info(
            f"✓ Vector database created with {len(self.chunks)} documents")

    @weave.op()
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant documents for a query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with metadata
        """
        if self.collection is None:
            raise ValueError(
                "Vector database not initialized. Call create_vector_database() first.")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query, convert_to_numpy=True)

        # Search in vector database
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return retrieved_docs

    @weave.op()
    def create_raft_prompt(self, query: str, documents: List[Dict]) -> str:
        """
        Create RAFT-style prompt with retrieved documents

        Args:
            query: User question
            documents: Retrieved documents

        Returns:
            Formatted prompt for RAFT model
        """
        # Format documents
        formatted_docs = "\n\n".join([
            f"Document [{i+1}] ({doc['metadata'].get('title', 'Untitled')}):\n{doc['text']}"
            for i, doc in enumerate(documents)
        ])

        # Create RAFT prompt
        prompt = f"""Use the following documents to answer the question. If the answer cannot be found in the documents, say "I cannot answer based on the provided context."

{formatted_docs}

Question: {query}
Answer:"""

        return prompt

    @weave.op()
    def generate_answer(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """
        Generate answer using RAFT model

        Args:
            prompt: Formatted prompt with context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling

        Returns:
            Generated answer
        """
        if self.raft_model is None:
            return "⚠️  No RAFT model loaded. Load a model with load_raft_model() or train one with run_raft.py"

        # Tokenize
        inputs = self.raft_tokenizer.encode(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        # Generate
        with torch.no_grad():
            outputs = self.raft_model.generate(
                inputs,
                max_length=inputs.shape[1] + max_new_tokens,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.raft_tokenizer.eos_token_id,
                top_p=top_p,
                repetition_penalty=1.1
            )

        # Decode
        generated_text = self.raft_tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        # Extract answer (after "Answer:")
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()

        return answer

    @weave.op()
    def query(
        self,
        question: str,
        top_k: int = 3,
        show_sources: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Complete RAG query: retrieve + generate

        Args:
            question: User question
            top_k: Number of documents to retrieve
            show_sources: Include source documents in response
            verbose: Print detailed information

        Returns:
            Dictionary with answer and metadata
        """
        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"Query: {question}")
            logger.info(f"{'='*80}")

        # Step 1: Retrieve relevant documents
        if verbose:
            logger.info("\n[1/3] Retrieving relevant documents...")

        retrieved_docs = self.retrieve(question, top_k=top_k)

        if verbose:
            logger.info(f"✓ Retrieved {len(retrieved_docs)} documents")
            for i, doc in enumerate(retrieved_docs, 1):
                logger.info(f"  {i}. {doc['metadata'].get('title', 'Untitled')} "
                            f"(distance: {doc.get('distance', 'N/A'):.4f})")

        # Step 2: Create RAFT prompt
        if verbose:
            logger.info("\n[2/3] Creating RAFT prompt...")

        prompt = self.create_raft_prompt(question, retrieved_docs)

        if verbose:
            logger.info(f"✓ Prompt created ({len(prompt)} chars)")

        # Step 3: Generate answer
        if verbose:
            logger.info("\n[3/3] Generating answer with RAFT model...")

        answer = self.generate_answer(prompt)

        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info("Answer:")
            logger.info(f"{'='*80}")
            logger.info(answer)

            if show_sources:
                logger.info(f"\n{'='*80}")
                logger.info("Sources:")
                logger.info(f"{'='*80}")
                for i, doc in enumerate(retrieved_docs, 1):
                    logger.info(
                        f"\n[{i}] {doc['metadata'].get('title', 'Untitled')}")
                    logger.info(f"{doc['text'][:200]}...")

            logger.info(f"\n{'='*80}")

        return {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs,
            'prompt': prompt
        }

    def interactive_mode(self):
        """Interactive query mode"""
        logger.info("\n" + "="*80)
        logger.info("RAG SYSTEM - Interactive Mode")
        logger.info("="*80)
        logger.info(
            "Ask questions about Roberto Arce's professional background.")
        logger.info("Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                question = input("💬 Your question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    logger.info("Goodbye! 👋")
                    break

                if not question:
                    continue

                # Process query
                result = self.query(question, verbose=True)
                print()  # Extra newline for readability

            except KeyboardInterrupt:
                logger.info("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def find_latest_raft_model() -> Optional[str]:
    """Find the most recently created RAFT model"""
    model_dirs = list(Path(".").glob("finetuned_roberto_raft_*"))

    if not model_dirs:
        return None

    # Sort by modification time
    latest = max(model_dirs, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    """Main function to demonstrate RAG system"""

    print("="*80)
    print("RAG (Retrieval Augmented Generation) System")
    print("="*80)
    print()

    # Check dependencies
    if not CHROMADB_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ Missing dependencies. Please install:")
        print("   pip install chromadb sentence-transformers")
        return

    # Find RAFT model
    raft_model_path = find_latest_raft_model()

    if raft_model_path:
        print(f"✓ Found RAFT model: {raft_model_path}")
        use_raft = input("Use this RAFT model? (y/n): ").strip().lower() == 'y'
        if not use_raft:
            raft_model_path = None
    else:
        print("⚠️  No RAFT model found. You can:")
        print("   1. Run: python run_raft.py  (to train a RAFT model)")
        print("   2. Continue without RAFT (basic RAG only)")
        print()
        continue_anyway = input(
            "Continue without RAFT model? (y/n): ").strip().lower() == 'y'
        if not continue_anyway:
            return

    print()

    # Initialize RAG system
    rag = RAGSystem(
        source_file="roberto_data.txt",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        raft_model_path=raft_model_path,
        chunk_size=300
    )

    # Create vector database
    rag.create_vector_database()

    print()
    print("="*80)
    print("RAG System Ready! 🚀")
    print("="*80)
    print()

    # Test with example queries
    print("Running example queries...\n")

    example_queries = [
        "Where does Roberto work and what is his role?",
        "What programming languages and frameworks does Roberto know?",
        "What is Roberto's educational background?",
    ]

    for query in example_queries:
        rag.query(query, top_k=3, show_sources=False, verbose=True)
        print()

    # Interactive mode
    print("\n" + "="*80)
    response = input(
        "Would you like to try interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        rag.interactive_mode()

    print("\n✓ RAG system demonstration complete!")


if __name__ == "__main__":
    main()
