"""
Test script for RAG system with provided test questions
"""

import pandas as pd
from rag_system import RAGSystem
import os


def test_with_csv(rag: RAGSystem, csv_path: str, output_path: str = "test_results.csv"):
    """
    Test RAG system with questions from CSV file
    
    Args:
        rag: Initialized RAG system
        csv_path: Path to CSV with test questions
        output_path: Path to save results (default: test_results.csv)
    """
    # Load test questions
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*80}")
    print(f"Testing with {len(df)} questions from {csv_path}")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, row in df.iterrows():
        question = row['question']
        
        print(f"\n[{idx+1}/{len(df)}] Question: {question}")
        print("-" * 80)
        
        # Get answer from RAG system
        result = rag.answer_question(question, top_k=5, verbose=False)
        answer = result['answer']
        
        print(f"Answer: {answer}")
        print("=" * 80)
        
        # Save only question and answer (same format as test_set.csv)
        results.append({
            'question': question,
            'answer': answer
        })
    
    # Save results in same format as test_set.csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Results saved to {output_path}")
    print(f"Format: question,answer (same as {csv_path})")
    
    return results_df


def main():
    """Main test function"""
    print("🚀 Initializing RAG system...")
    print("⚠️  First run may take several minutes (downloading models)")
    
    # Initialize RAG system - will auto-load index if exists
    rag = RAGSystem(
        pdf_path="data/strategy.pdf",
        index_path="data/faiss_index",
        auto_load=True
    )
    
    # Build index only if not loaded
    if rag.index is None:
        print("\n🔨 Index not found, building new one...")
        rag.build_index(save_after_build=True)
        print("✓ Index created and saved")
    
    # Test with CSV questions
    test_with_csv(rag, "data/test_set.csv", "test_results.csv")
    
    print("\n" + "="*80)
    print("Testing complete! Check test_results.csv for detailed results.")
    print("="*80)


if __name__ == "__main__":
    main()