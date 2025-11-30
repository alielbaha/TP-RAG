# TP-RAG

A simple end-to-end RAG (Retrieval-Augmented Generation) pipeline demonstrating document ingestion, vector indexing, semantic retrieval, and LLM-based answer generation.

This repository is designed for learning, experimentation, and prototyping small RAG systems.



---

## Installation

```bash
git clone https://github.com/alielbaha/TP-RAG
cd TP-RAG
pip install -r requirements.txt
```

## Features
  
- Embedding generation & vector storage  
- Semantic retrieval (top-k)  
- RAG pipeline combining retrieved chunks + LLM  
- Streamlit interface

---

## Structure
rag-project/

├── src/

│   ├── __init__.py

│   ├── indexer.py # Q1

│   ├── retriever.py  # Q2

│   ├── llm_handler.py # Q3

│   ├── evaluator.py # Q4

│   └── utils.py

├── data/

│   ├── document1.pdf  (le fameux livre "intrroduction to statistical learning")
 
├── cli.py

├── config.yaml

├── app.py (streamlit)

├── requirements.txt

└── report.pdf



## Modèles
### Embedding:
sentence-transformers/all-MiniLM-L6-v2

### QA :
Llm : zephyr-7b-betazephyr-7b-beta
