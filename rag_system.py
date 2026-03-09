"""
RAG System for Russian AI Strategy Document
Uses Hugging Face models for both embeddings and generation
"""

import os
import re
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PyPDF2 import PdfReader
from tqdm import tqdm
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGSystem:
    """RAG system for answering questions about AI strategy document"""
    
    def __init__(self, pdf_path: str,
                 index_path: str = "data/faiss_index",
                 auto_load: bool = True):
        """
        Initialize RAG system
        
        Args:
            pdf_path: Path to PDF document
            index_path: Path to save/load FAISS index
            auto_load: If True, automatically load index if it exists
        """
        self.pdf_path = pdf_path
        self.chunk_size = 800
        self.chunk_overlap = 200
        self.index_path = index_path
        self.llm_model = "Qwen/Qwen2.5-1.5B-Instruct"
        
        # Initialize embedding model (multilingual model for Russian text)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        # Initialize LLM from Hugging Face
        print(f"Loading LLM model: {self.llm_model}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load text generation model and tokenizer
        print(f"Loading text generation model: {self.llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Use pipeline for text generation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        print("✓ Models loaded successfully")
        
        # Storage for chunks and embeddings
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.index = None
        
        # Auto-load index if it exists
        if auto_load and os.path.exists(f"{self.index_path}.index"):
            print(f"\n📂 Found existing index at {self.index_path}")
            try:
                self.load_index(self.index_path)
                print("✓ Index loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load index: {e}")
                print("Will need to build new index")
        
    def extract_text_from_pdf(self) -> str:
        """Extract text from PDF file"""
        print(f"Extracting text from {self.pdf_path}...")
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and artifacts
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()
    
    def create_chunks(self, text: str) -> List[Tuple[str, Dict]]:
        """
        Split text into chunks using RecursiveCharacterTextSplitter
        
        Args:
            text: Full document text
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        print("Creating chunks with RecursiveCharacterTextSplitter...")
        
        # Initialize RecursiveCharacterTextSplitter
        # It tries to split on paragraphs, then sentences, then words
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            is_separator_regex=False
        )
        
        # Split text into chunks
        chunk_texts = text_splitter.split_text(text)
        
        # Create chunks with metadata
        chunks = []
        for chunk_id, chunk_text in enumerate(chunk_texts):
            chunk_text = chunk_text.strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append((
                    chunk_text,
                    {
                        "chunk_id": chunk_id,
                        "length": len(chunk_text),
                        "type": "recursive"
                    }
                ))
        
        print(f"Created {len(chunks)} chunks from {len(text)} characters")
        if chunks:
            avg_length = sum(c[1]['length'] for c in chunks) / len(chunks)
            print(f"Average chunk size: {avg_length:.0f} characters")
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks
        
        Args:
            texts: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        print("Creating embeddings...")
        # Add instruction prefix for better retrieval (E5 model requirement)
        texts_with_prefix = [f"passage: {text}" for text in texts]
        embeddings = self.embedding_model.encode(
            texts_with_prefix,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Verify embeddings are valid
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("⚠️  Warning: Invalid values detected in embeddings, cleaning...")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return embeddings
    
    def build_index(self, save_after_build: bool = True):
        """
        Build FAISS index from document
        
        Args:
            save_after_build: If True, automatically save index after building
        """
        # Extract and process text
        raw_text = self.extract_text_from_pdf()
        clean_text = self.clean_text(raw_text)
        
        # Create chunks
        chunk_data = self.create_chunks(clean_text)
        self.chunks = [chunk[0] for chunk in chunk_data]
        self.chunk_metadata = [chunk[1] for chunk in chunk_data]
        
        # Create embeddings
        embeddings = self.create_embeddings(self.chunks)
        
        # Build FAISS index using L2 distance (more stable than inner product)
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use L2 (Euclidean) distance - more stable and intuitive
        # Lower distance = more similar
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
        print(f"Embedding dimension: {dimension}")
        
        # Auto-save if requested
        if save_after_build:
            self.save_index(self.index_path)
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk_text, score) tuples sorted by relevance (lowest distance first)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Create query embedding with instruction prefix
        query_with_prefix = f"query: {query}"
        query_embedding = self.embedding_model.encode(
            [query_with_prefix],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Verify query embedding is valid
        if np.any(np.isnan(query_embedding)) or np.any(np.isinf(query_embedding)):
            print("⚠️  Warning: Invalid query embedding, cleaning...")
            query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Search in FAISS index
        # Using L2 distance: lower distance = more similar
        # Distances are always >= 0
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Convert distances to similarity scores (inverse)
        # similarity = 1 / (1 + distance)
        # This gives scores between 0 and 1, where 1 is most similar
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                similarity = 1.0 / (1.0 + float(distance))
                results.append((self.chunks[idx], similarity))
        
        return results
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer using text generation pipeline
        
        Args:
            query: User question
            context_chunks: Relevant text chunks
            
        Returns:
            Generated answer
        """
        try:
            # Combine top chunks for context
            # Llama 3.2 has larger context window (128k tokens), but we'll be conservative
            combined_context = "\n\n".join(context_chunks[:3])
            
            # Truncate context if too long
            max_context_length = 2000  # characters
            if len(combined_context) > max_context_length:
                combined_context = combined_context[:max_context_length] + "..."
            
            # Create prompt for Qwen (instruction format)
            prompt = f"""Ты помощник, который отвечает на вопросы на основе предоставленного контекста. Отвечай кратко и по существу только на русском языке. Названия правовых актов и документов оставляй без изменений. Не выдумывай информацию, которой нет в контексте, даже если об этом явно просят в вопросе. Если в контексте нет ответа на вопрос, то не выдумывай ответ и прямо напиши в ответе, что информация в контексте отсутствует. Не используй markdown или иную разметку в ответе. Не используй переносы строк в ответе. <|.im_end|>
<|im_start|>user
Контекст: {combined_context}

Вопрос: {query}<|im_end|>
<|im_start|>assistant
"""
            
            # Generate answer
            result = self.generator(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text (remove prompt)
            generated_text = result[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            # Remove any trailing special tokens
            answer = answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            
            # If answer is empty or too short, return context snippet
            if not answer or len(answer) < 10:
                snippet = context_chunks[0][:400].strip()
                return f"Согласно документу: {snippet}..."
            
            return answer
        
        except Exception as e:
            return f"Ошибка при генерации ответа: {str(e)}"
    
    def answer_question(self, question: str, top_k: int = 5, verbose: bool = False) -> Dict:
        """
        Answer a question using RAG pipeline
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            verbose: Whether to print debug information
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print(f"{'='*80}")
            print(f"\nRetrieved {len(relevant_chunks)} chunks:")
            for i, (chunk, score) in enumerate(relevant_chunks, 1):
                print(f"\nChunk {i} (score: {score:.4f}):")
                print(f"{chunk[:200]}...")
            print(f"\n{'='*80}\n")
        
        # Extract just the text chunks
        context_chunks = [chunk for chunk, _ in relevant_chunks]
        
        # Generate answer
        answer = self.generate_answer(question, context_chunks)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": len(relevant_chunks),
            "relevance_scores": [score for _, score in relevant_chunks]
        }
    
    def save_index(self, index_path: str = "data/faiss_index"):
        """Save FAISS index and chunks to disk"""
        if self.index is None:
            raise ValueError("Index not built yet")
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, f"{index_path}.index")
        
        # Save chunks and metadata
        np.savez(
            f"{index_path}_data.npz",
            chunks=np.array(self.chunks, dtype=object),
            metadata=np.array(self.chunk_metadata, dtype=object)
        )
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str = "data/faiss_index"):
        """Load FAISS index and chunks from disk"""
        self.index = faiss.read_index(f"{index_path}.index")
        
        # Load chunks and metadata
        data = np.load(f"{index_path}_data.npz", allow_pickle=True)
        self.chunks = data['chunks'].tolist()
        self.chunk_metadata = data['metadata'].tolist()
        
        print(f"Index loaded from {index_path}")


def main():
    """Main function to demonstrate RAG system"""
    print("\n" + "="*80)
    print("RAG System Demo")
    print("="*80 + "\n")
    
    # Initialize RAG system - will auto-load index if exists
    rag = RAGSystem(
        pdf_path="data/strategy.pdf",
        index_path="data/faiss_index",
        auto_load=True
    )
    
    # Build index only if not loaded
    if rag.index is None:
        print("\n🔨 Building new index...")
        rag.build_index(save_after_build=True)
        print("✓ Index created and saved")
    
    # Test with sample questions
    test_questions = [
        "Какие федеральные законы составляют правовую основу Стратегии?",
        "Что в Стратегии понимается под искусственным интеллектом?",
        "Какие показатели используются для оценки достижения целей Стратегии?"
    ]
    
    print("\n" + "="*80)
    print("Testing RAG System")
    print("="*80 + "\n")
    
    for question in test_questions:
        result = rag.answer_question(question, verbose=True)
        print(f"Answer: {result['answer']}\n")
        print("-"*80 + "\n")


if __name__ == "__main__":
    main()