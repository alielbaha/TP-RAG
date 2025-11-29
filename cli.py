#!/usr/bin/env python3
"""
Command Line Interface for RAG System
Supports document indexing, retrieval, and question-answering.

"""

import argparse
import sys
from pathlib import Path

from src.indexer import DocIndexer
from src.retriever import Retriever
from src.llm_handler import LLMQuestionAnswering
from src.utils import load_config, setup_logging, print_config_summary, ensure_directories


def cmd_index(args, config):
    """
    Index documents into the vector store.
    """
    print("\n" + "="*80)
    print("DOCUMENT INDEXING")
    print("="*80 + "\n")

    # Get configuration
    indexing_config = config['indexing']
    
    # Override with command line arguments if provided
    data_path = args.data_path or indexing_config['data_path']
    
    # Initialize indexer
    indexer = DocIndexer(
        embedding_model_name=indexing_config['embedding_model'],
        vector_store_path=indexing_config['vector_store']['path'],
        collection_name=indexing_config['vector_store']['collection_name'],
        chunk_size=indexing_config['text_splitting']['chunk_size'],
        chunk_overlap=indexing_config['text_splitting']['chunk_overlap'],
        use_markdown_splitter=indexing_config['text_splitting']['use_markdown_splitter']
    )

    # Index documents
    try:
        print(f"Indexing documents from: {data_path}")
        vector_store = indexer.index_documents(
            data_path=data_path,
            file_type=indexing_config['file_type'],
            persist=True
        )

        # Display statistics
        stats = indexer.get_stats()
        print("\n" + "="*80)
        print("INDEXING STATISTICS")
        print("="*80)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✓ Indexing completed successfully!\n")

    except Exception as e:
        print(f"\n✗ Error during indexing: {str(e)}\n")
        sys.exit(1)


def cmd_search(args, config):
    """
    Search the vector database with a query.
    """
    print("\n" + "="*80)
    print("DOCUMENT SEARCH")
    print("="*80 + "\n")

    # Get configuration
    indexing_config = config['indexing']
    retrieval_config = config['retrieval']

    # Initialize retriever
    retriever = Retriever(
        vector_store_path=indexing_config['vector_store']['path'],
        embedding_model_name=indexing_config['embedding_model'],
        collection_name=indexing_config['vector_store']['collection_name'],
        top_k=args.top_k or retrieval_config['top_k'],
        search_type=retrieval_config['search_type'],
        score_threshold=retrieval_config['score_threshold']
    )

    # Load vector store
    try:
        retriever.load_vector_store()
    except Exception as e:
        print(f"✗ Error loading vector store: {str(e)}")
        print("  Have you indexed documents yet? Run: python cli.py index --data-path ./data")
        sys.exit(1)

    # Perform search
    query = args.query
    print(f"Query: '{query}'\n")

    try:
        results = retriever.retrieve_documents(
            query=query,
            top_k=args.top_k or retrieval_config['top_k'],
            return_scores=True
        )

        # Display results
        retriever.print_search_results(
            query=query,
            results=results,
            max_content_length=args.max_length or 300
        )

        if args.show_context:
            print("\n" + "="*80)
            print("CONCATENATED CONTEXT FOR LLM")
            print("="*80 + "\n")
            context = retriever.get_relevant_context(query, top_k=args.top_k)
            print(context)
            print("\n")

    except Exception as e:
        print(f"\n✗ Error during search: {str(e)}\n")
        sys.exit(1)


def cmd_list_sources(args, config):
    """
    List all document sources in the vector store.
    """
    print("\n" + "="*80)
    print("INDEXED DOCUMENTS")
    print("="*80 + "\n")

    # Get configuration
    indexing_config = config['indexing']

    # Initialize retriever
    retriever = Retriever(
        vector_store_path=indexing_config['vector_store']['path'],
        embedding_model_name=indexing_config['embedding_model'],
        collection_name=indexing_config['vector_store']['collection_name']
    )

    try:
        retriever.load_vector_store()
        sources = retriever.get_unique_sources()

        if not sources:
            print("No documents found in the vector store.")
            print("Run: python cli.py index --data-path ./data")
        else:
            print(f"Found {len(sources)} unique document(s):\n")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {Path(source).name}")
                if args.verbose:
                    print(f"     Path: {source}")
        
        print()

    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
        sys.exit(1)


def cmd_ask(args, config):
    """
    Ask a question and get an answer using the RAG system.
    """
    print("\n" + "="*80)
    print("RAG QUESTION-ANSWERING")
    print("="*80 + "\n")

    # Get configuration
    indexing_config = config['indexing']
    retrieval_config = config['retrieval']
    llm_config = config['llm']

    # Initialize retriever
    print("Initializing retriever...")
    retriever = Retriever(
        vector_store_path=indexing_config['vector_store']['path'],
        embedding_model_name=indexing_config['embedding_model'],
        collection_name=indexing_config['vector_store']['collection_name'],
        top_k=args.top_k or retrieval_config['top_k']
    )

    try:
        retriever.load_vector_store()
    except Exception as e:
        print(f"✗ Error loading vector store: {str(e)}")
        print("  Run: python cli.py index --data-path ./data")
        sys.exit(1)

    # Initialize LLM
    print(f"Loading LLM: {llm_config['model_name']}...")
    print("(This may take a minute on first run)\n")
    
    llm_handler = LLMQuestionAnswering(
        model_name=llm_config['model_name'],
        device=llm_config['device'],
        max_new_tokens=llm_config['max_new_tokens'],
        temperature=llm_config['temperature'],
        top_p=llm_config.get('top_p', 0.95),
        do_sample=llm_config.get('do_sample', True),
        use_api=llm_config.get('use_api', False),
        api_token=llm_config.get('api_token')
    )

    try:
        llm_handler.load_model()
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        print("  Consider using 'google/flan-t5-base' or enable API mode")
        sys.exit(1)

    # Create prompt template
    prompt_template = llm_handler.create_prompt_template(
        template=config['prompt']['template']
    )

    # Process question
    question = args.question
    print(f"Question: {question}\n")
    print("Generating answer...\n")

    try:
        result = llm_handler.answer_with_retrieval(
            question=question,
            retriever=retriever,
            prompt_template=prompt_template,
            top_k=args.top_k or retrieval_config['top_k']
        )

        # Display answer
        print("="*80)
        print("ANSWER")
        print("="*80)
        print(f"\n{result['answer']}\n")

        # Display sources
        if not args.no_sources:
            print("="*80)
            print("SOURCES")
            print("="*80)
            for i, source in enumerate(result['sources'], 1):
                source_name = Path(source['source']).name
                print(f"\n{i}. {source_name} (Page {source['page']})")
                print(f"   Relevance Score: {source['score']:.4f}")
                if args.show_context:
                    print(f"   Context: {source['content_preview']}...")

        # Display metadata
        if args.verbose:
            print("\n" + "="*80)
            print("METADATA")
            print("="*80)
            print(f"Model: {result['model']}")
            print(f"Context Length: {result['context_length']} characters")
            print(f"Number of Sources: {result['num_sources']}")

        print()

    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")
        sys.exit(1)


def cmd_interactive(args, config):
    """
    Interactive question-answering mode.
    """
    print("\n" + "="*80)
    print("INTERACTIVE RAG QA MODE")
    print("="*80)
    print("\nType 'quit' or 'exit' to stop.")
    print("Type 'help' for available commands.\n")

    # Get configuration
    indexing_config = config['indexing']
    retrieval_config = config['retrieval']
    llm_config = config['llm']

    # Initialize components
    print("Initializing system...")
    
    retriever = Retriever(
        vector_store_path=indexing_config['vector_store']['path'],
        embedding_model_name=indexing_config['embedding_model'],
        collection_name=indexing_config['vector_store']['collection_name'],
        top_k=retrieval_config['top_k']
    )

    try:
        retriever.load_vector_store()
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        sys.exit(1)

    print("Loading LLM (this may take a minute)...")
    
    llm_handler = LLMQuestionAnswering(
        model_name=llm_config['model_name'],
        device=llm_config['device'],
        max_new_tokens=llm_config['max_new_tokens'],
        temperature=llm_config['temperature'],
        use_api=llm_config.get('use_api', False),
        api_token=llm_config.get('api_token')
    )

    try:
        llm_handler.load_model()
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        sys.exit(1)

    prompt_template = llm_handler.create_prompt_template(
        template=config['prompt']['template']
    )

    print("✓ System ready!\n")

    # Interactive loop
    while True:
        try:
            question = input("❓ Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if question.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Type any question to get an answer")
                print("  - 'sources' - List indexed documents")
                print("  - 'quit' or 'exit' - Exit interactive mode")
                print()
                continue

            if question.lower() == 'sources':
                sources = retriever.get_unique_sources()
                print(f"\nIndexed documents ({len(sources)}):")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {Path(source).name}")
                print()
                continue

            if not question:
                continue

            print("\n💭 Thinking...\n")

            result = llm_handler.answer_with_retrieval(
                question=question,
                retriever=retriever,
                prompt_template=prompt_template,
                top_k=retrieval_config['top_k']
            )

            print(f"💡 Answer:\n{result['answer']}\n")

            print(f"📚 Sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                source_name = Path(source['source']).name
                print(f"   {i}. {source_name} (Page {source['page']})")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {str(e)}\n")
    """
    Run test queries on the indexed documents.
    """
    print("\n" + "="*80)
    print("TESTING RETRIEVAL SYSTEM")
    print("="*80 + "\n")

    # Get configuration
    indexing_config = config['indexing']
    retrieval_config = config['retrieval']

    # Initialize retriever
    retriever = Retriever(
        vector_store_path=indexing_config['vector_store']['path'],
        embedding_model_name=indexing_config['embedding_model'],
        collection_name=indexing_config['vector_store']['collection_name'],
        top_k=retrieval_config['top_k'],
        search_type=retrieval_config['search_type'],
        score_threshold=retrieval_config['score_threshold']
    )

    try:
        retriever.load_vector_store()

        # Default test queries
        test_queries = [
            "What is the main topic discussed?",
            "What are the key findings?",
            "What methodology is used?"
        ]

        print(f"Running {len(test_queries)} test queries...\n")

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'─'*80}")
            print(f"TEST {i}/{len(test_queries)}")
            print(f"{'─'*80}")
            
            results = retriever.retrieve_documents(query, top_k=3)
            retriever.print_search_results(query, results, max_content_length=200)

        # Evaluation
        eval_results = retriever.evaluate_retrieval(test_queries)
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"  Queries with results: {eval_results['queries_with_results']}/{eval_results['total_queries']}")
        print(f"  Average score: {eval_results['average_score']:.4f}")
        print(f"  Average results per query: {eval_results['average_results_per_query']:.2f}")
        print()

    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}\n")
        sys.exit(1)


def main():
    """
    Main CLI entry point.
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        description="RAG System - Document Indexing and Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents
  python cli.py index --data-path ./data

  # Search documents
  python cli.py search "What are the main findings?"

  # Ask a question (RAG QA)
  python cli.py ask "What is the methodology used?"

  # Interactive mode
  python cli.py interactive

  # List indexed documents
  python cli.py list

  # Run test queries
  python cli.py test
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Index command
    parser_index = subparsers.add_parser('index', help='Index documents into vector store')
    parser_index.add_argument(
        '--data-path',
        type=str,
        help='Path to documents directory or file'
    )

    # Search command
    parser_search = subparsers.add_parser('search', help='Search the vector database')
    parser_search.add_argument(
        'query',
        type=str,
        help='Search query'
    )
    parser_search.add_argument(
        '--top-k',
        type=int,
        help='Number of results to return'
    )
    parser_search.add_argument(
        '--max-length',
        type=int,
        help='Maximum content length to display'
    )
    parser_search.add_argument(
        '--show-context',
        action='store_true',
        help='Show concatenated context for LLM'
    )

    # List sources command
    parser_list = subparsers.add_parser('list', help='List indexed documents')
    parser_list.add_argument(
        '--verbose',
        action='store_true',
        help='Show full paths'
    )

    # Test command
    parser_test = subparsers.add_parser('test', help='Run test queries')

    # Ask command (Q3)
    parser_ask = subparsers.add_parser('ask', help='Ask a question using RAG')
    parser_ask.add_argument(
        'question',
        type=str,
        help='Question to ask'
    )
    parser_ask.add_argument(
        '--top-k',
        type=int,
        help='Number of context documents to use'
    )
    parser_ask.add_argument(
        '--no-sources',
        action='store_true',
        help='Do not show source documents'
    )
    parser_ask.add_argument(
        '--show-context',
        action='store_true',
        help='Show context snippets from sources'
    )
    parser_ask.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed metadata'
    )

    # Interactive command (Q3)
    parser_interactive = subparsers.add_parser('interactive', help='Interactive QA mode')

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command provided
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Load configuration
    try:
        config = load_config(args.config)
        
        # Setup logging
        log_config = config.get('logging', {})
        setup_logging(
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file'),
            console_output=log_config.get('console_output', True)
        )

        # Ensure required directories exist
        ensure_directories(config)

    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)

    # Execute command
    commands = {
        'index': cmd_index,
        'search': cmd_search,
        'list': cmd_list_sources,
        'ask': cmd_ask,
        'interactive': cmd_interactive
    }

    if args.command in commands:
        commands[args.command](args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()