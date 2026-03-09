"""
Interactive RAG system for querying the AI Strategy document
"""

import os
from rag_system import RAGSystem


def print_header():
    """Print application header"""
    print("\n" + "="*80)
    print("RAG Система для Национальной Стратегии Развития ИИ до 2030 года")
    print("="*80)
    print("\nИспользует Hugging Face модели (DeBERTa QA) для ответов на вопросы")
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'help' для справки\n")


def print_help():
    """Print help information"""
    print("\n" + "-"*80)
    print("СПРАВКА:")
    print("-"*80)
    print("• Задавайте вопросы о содержании документа")
    print("• Система найдет релевантные фрагменты и сгенерирует ответ")
    print("• Примеры вопросов:")
    print("  - Какие цели развития ИИ указаны в стратегии?")
    print("  - Что такое большие фундаментальные модели?")
    print("  - Какие показатели используются для оценки достижения целей?")
    print("-"*80 + "\n")


def interactive_mode(rag: RAGSystem):
    """
    Run interactive question-answering mode
    
    Args:
        rag: Initialized RAG system
    """
    print_header()
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Ваш вопрос: ").strip()
            
            # Check for exit commands
            if question.lower() in ['exit', 'quit', 'выход']:
                print("\n👋 До свидания!")
                break
            
            # Check for help
            if question.lower() in ['help', 'помощь', 'справка']:
                print_help()
                continue
            
            # Skip empty questions
            if not question:
                continue
            
            # Process question
            print("\n🔍 Поиск релевантных фрагментов...")
            result = rag.answer_question(question, top_k=5, verbose=False)
            
            # Display answer
            print("\n" + "="*80)
            print("💡 ОТВЕТ:")
            print("="*80)
            print(f"\n{result['answer']}\n")
            
            # Display metadata
            avg_score = sum(result['relevance_scores']) / len(result['relevance_scores'])
            print("-"*80)
            print(f"📊 Использовано фрагментов: {result['retrieved_chunks']}")
            print(f"📈 Средняя релевантность: {avg_score:.4f}")
            print("-"*80)
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")


def main():
    """Main function"""
    print("\n🚀 Инициализация RAG системы...")
    print("⚠️  Первый запуск может занять несколько минут (загрузка моделей)")
    
    # Initialize RAG system - will auto-load index if exists
    rag = RAGSystem(
        pdf_path="data/strategy.pdf",
        index_path="data/faiss_index",
        auto_load=True
    )
    
    # Build index only if not loaded
    if rag.index is None:
        print("\n🔨 Индекс не найден, создаем новый...")
        rag.build_index(save_after_build=True)
        print("✓ Индекс создан и сохранен")
    
    # Start interactive mode
    interactive_mode(rag)


if __name__ == "__main__":
    main()