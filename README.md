
End-to-end RAG 


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

```
.
├── chroma_db/
├── data/
├── demo/  
├── src/
│   ├── indexer.py      
│   ├── retriever.py     
│   ├── llm_handler.py      
│   ├── evaluator.py      
│   └── utils.py          
├── app.py       
├── __init__.py             
├── config.yaml             
└── readme.md                 
```


## Modèles
### Embedding:
sentence-transformers/all-MiniLM-L6-v2

### QA :
Llm : zephyr-7b-betazephyr-7b-beta
