
import os
import logging
import pandas as pd
from typing import List, Dict, Any
from src.retriever import Retriever
from src.llm_handler import LLMQuestionAnswering
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate as ragas_evaluate
from ragas.llms import LangchainLLM
from ragas.embeddings import LangchainEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(
        self,
        vector_store_path: str,
        embedding_model_name: str,
        llm_model_name: str,
        hf_api_token: str,
        collection_name: str = "documents"
    ):
        self.retriever = Retriever(
            vector_store_path=vector_store_path,
            embedding_model_name=embedding_model_name,
            collection_name=collection_name
        )

        self.llm_handler = LLMQuestionAnswering(
            model_name=llm_model_name,
            use_api=True,
            api_token=hf_api_token
        )

        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.vector_store_path = vector_store_path

    # -----
    def evaluate_retrieval(self, test_queries: List[str], expected_sources: List[List[str]] = None) -> Dict[str, Any]:
        logger.info("Evaluating retrieval performance...")
        return self.retriever.evaluate_retrieval(test_queries, expected_sources)

    def evaluate_generation(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate generation using RAGAS metrics.
        dataset: pandas DataFrame with columns — ['question', 'answer', 'contexts', 'ground_truth']
        """
        logger.info("Evaluating generation quality using RAGAS metrics...")

        ragas_dataset = Dataset.from_dict({
            "question": dataset["question"].tolist(),
            "answer": dataset["answer"].tolist(),
            "contexts": dataset["contexts"].tolist(),
            "ground_truth": dataset["ground_truth"].tolist()
        })

        lc_llm = LangchainLLM(self.llm_handler.llm)
        lc_emb = LangchainEmbeddings(HuggingFaceEmbeddings(model_name=self.embedding_model_name))

        result = ragas_evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=lc_llm,
            embeddings=lc_emb
        )

        return result.to_pandas().mean().to_dict()

    
    def evaluate_rag(self, test_set: List[Dict[str, str]], top_k: int = 3) -> pd.DataFrame:
        """
        Full RAG evaluation pipeline.
        test_set: list of dicts → {"question": str, "ground_truth": str}
        """
        logger.info("Running end-to-end RAG evaluation...")
        records = []

        for item in test_set:
            q = item["question"]
            gt = item["ground_truth"]

            result = self.llm_handler.answer_with_retrieval(q, self.retriever, top_k=top_k)
            records.append({
                "question": q,
                "ground_truth": gt,
                "answer": result["answer"],
                "contexts": [src["content_preview"] for src in result["sources"]],
                "num_sources": result["num_sources"],
                "model": result["model"]
            })

        return pd.DataFrame(records)
