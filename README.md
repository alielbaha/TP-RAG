# TP-RAG

A simple end-to-end RAG (Retrieval-Augmented Generation) pipeline demonstrating document ingestion, vector indexing, semantic retrieval, and LLM-based answer generation.

This repository is designed for learning, experimentation, and prototyping small RAG systems.



---

## ⚙️ Installation

```bash
git clone https://github.com/alielbaha/TP-RAG
cd TP-RAG
pip install -r requirements.txt
```

## 🚀 Features

- PDF ingestion & text extraction  
- Text chunking with metadata  
- Embedding generation & vector storage  
- Semantic retrieval (top-k)  
- RAG pipeline combining retrieved chunks + LLM  
- Simple evaluation module  
- CLI for easy interaction  

---

## 📁 Project Structure
rag-project/

├── src/

│   ├── __init__.py

│   ├── indexer.py # Q1

│   ├── retriever.py  # Q2

│   ├── llm_handler.py # Q3

│   ├── evaluator.py # Q4

│   └── utils.py

├── data/

│   ├── document1.pdf

│   ├── document2.pdf

│   └── document3.pdf

├── cli.py

├── config.yaml

├── requirements.txt

└── report.pdf
