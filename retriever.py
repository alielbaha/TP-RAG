""" Document Retrieval System for RAG """

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    """ A class to handle document retrieval using vector embeddings. """
    def __init__(
            self,
            vector_store_path: str,
            embedding_model_name: str,
            collection_name: str = "documents",
            top_k: int = 5,
            search_type: str = "similarity",
            score_threshold: float = 0.0
        ):

        self.vector_store_path = vector_store_path
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.top_k = top_k
        self.search_type = search_type
        self.score_threshold = score_threshold

        self.embedding = None
        self.vector_store = None

        logger.info("Initialisation du Retriever avec top_k={top_k}")

    def _initialize_embedding(self) -> HuggingFaceEmbeddings:
        if self.embedding is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embedding model loaded successfully.")
        return self.embedding

    def load_vector_store(self) -> Chroma:
        logger.info(f"Loading vector store from: {self.vector_store_path}")
        embeddings = self._initialize_embedding()

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.vector_store_path
        )

        collection = self.vector_store._collection
        count = collection.count()

        if count == 0:
            logger.warning("The vector store is empty. Please add documents before querying.")
        else:
            logger.info(f"Vector store loaded with {count} documents.")

        return self.vector_store
    
    def search(self, query:str, top_k:Optional[int]=None, filter_dict:Optional[Dict[str, Any]]=None) -> List[Tuple[Document, float]]:
        if self.vector_store is None:
            self.load_vector_store()

        K =  top_k if top_k is not None else self.top_k

        logger.info(f"Searching for: '{query}' with top_k={K}")

        try:
            if self.search_type == "similarity":
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=K,
                    filter=filter_dict
                )
            
            elif self.search_type == "mmr":
                documents = self.vector_store.max_marginal_relevance_search(
                    query = query,
                    k = K,
                    filter = filter_dict,
                    fetch_k = K * 2
                )
                results = []
                for doc in documents:
                    score = self._calculate_similarity(query, doc.page_content)
                    results.append((doc, score))
            else: 
                raise ValueError(f"Unsupported search type: {self.search_type}")
            
            filtered_results = [
                (doc, score) for doc, score in results if score >= self.score_threshold
            ]

            logger.info(f"Found {len(filtered_results)} results after applying score threshold of {self.score_threshold}.")
            return filtered_results
        
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def _calculate_similarity(self, query: str, text: str) -> float:
        if self.embedding is None:
            self._initialize_embedding()
        
        import numpy as np

        query_embedding = self.embedding.embed_query(query)
        text_embedding = self.embedding.embed_query(text)

        # cosine similarity calculation
        similarity = np.dot(query_embedding, text_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
        )

        return float((similarity+1)/2)
    

    def retrieve_documents(
            self,
            query: str,
            top_k: Optional[int] = None,
            return_scores: bool = True
        ) -> List[Dict[str, Any]]:
        results = self.search(query, top_k=top_k)

        formatted_results = []
        for i, (doc, score) in enumerate(results, start=1):

            result = {
                "rank": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "N/A")
            }

            if return_scores:
                result["similarity_score"] = round(score, 4)
            
            formatted_results.append(result)
        return formatted_results
    

    def get_relevant_context(
            self,
            query: str,
            top_k: Optional[int] = None,
            separator: str = "\n\n---\n\n"
        ) -> str:
        results = self.retrieve_documents(query, top_k=top_k, return_scores=False)

        contexts = []
        for result in results:
            source_info = f"[Source: {Path(result['source']).name}, Page: {result['page']}]"
            context = f"{source_info}\n{result['content']}"
            contexts.append(context)

        return separator.join(contexts)
    

    def search_by_metadata(
            self,
            metadata_filter: Dict[str, Any],
            top_k: Optional[int] = None
        ) -> List[Document]:
        if self.vector_store is None:
            self.load_vector_store()
        
        K = top_k if top_k is not None else self.top_k
        logger.info(f"Searching by metadata filter: {metadata_filter} with top_k={K}")

        # get all documents matching the metadata filter
        collection = self.vector_store._collection
        results = collection.get(
            where=metadata_filter,
            limit=K
        )

        documents = []
        if results['documents']:
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                documents.append(Document(page_content=doc, metadata=metadata))

        logger.info(f"Found {len(documents)} documents matching metadata")
        return documents

