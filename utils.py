"""
Utility functions for the RAG project
"""

import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output logs to console

    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['indexing', 'retrieval', 'llm', 'prompt']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate indexing config
    indexing = config['indexing']
    if 'embedding_model' not in indexing:
        raise ValueError("Missing 'embedding_model' in indexing configuration")
    
    # Validate chunk sizes
    chunk_size = indexing.get('text_splitting', {}).get('chunk_size', 1000)
    chunk_overlap = indexing.get('text_splitting', {}).get('chunk_overlap', 200)
    
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    # Validate retrieval config
    retrieval = config['retrieval']
    if 'top_k' not in retrieval:
        raise ValueError("Missing 'top_k' in retrieval configuration")
    
    if retrieval['top_k'] < 1:
        raise ValueError("top_k must be at least 1")

    # Validate LLM config
    llm = config['llm']
    if 'model_name' not in llm:
        raise ValueError("Missing 'model_name' in llm configuration")


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all required directories exist.

    Args:
        config: Configuration dictionary
    """
    # Create data directory
    data_path = config['indexing'].get('data_path', './data')
    os.makedirs(data_path, exist_ok=True)

    # Create vector store directory
    vector_store_path = config['indexing']['vector_store']['path']
    os.makedirs(vector_store_path, exist_ok=True)

    # Create logs directory if logging to file
    if 'logging' in config and 'log_file' in config['logging']:
        log_file = config['logging']['log_file']
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)


def format_document_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format document metadata for display.

    Args:
        metadata: Document metadata dictionary

    Returns:
        Formatted metadata string
    """
    formatted = []
    
    if 'source' in metadata:
        source = Path(metadata['source']).name
        formatted.append(f"Source: {source}")
    
    if 'page' in metadata:
        formatted.append(f"Page: {metadata['page']}")
    
    if 'chunk_id' in metadata:
        formatted.append(f"Chunk: {metadata['chunk_id']}")

    return " | ".join(formatted) if formatted else "No metadata"


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("RAG SYSTEM CONFIGURATION SUMMARY")
    print("="*60)
    
    print("\n[INDEXING]")
    print(f"  Embedding Model: {config['indexing']['embedding_model']}")
    print(f"  Vector Store: {config['indexing']['vector_store']['path']}")
    print(f"  Chunk Size: {config['indexing']['text_splitting']['chunk_size']}")
    print(f"  Chunk Overlap: {config['indexing']['text_splitting']['chunk_overlap']}")
    
    print("\n[RETRIEVAL]")
    print(f"  Top K: {config['retrieval']['top_k']}")
    print(f"  Search Type: {config['retrieval']['search_type']}")
    
    print("\n[LLM]")
    print(f"  Model: {config['llm']['model_name']}")
    print(f"  Max Tokens: {config['llm']['max_new_tokens']}")
    print(f"  Temperature: {config['llm']['temperature']}")
    print(f"  Device: {config['llm']['device']}")
    
    print("\n" + "="*60 + "\n")

