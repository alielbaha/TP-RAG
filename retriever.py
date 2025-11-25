""" Document Retrieval System for RAG """

import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document



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

    def get_unique_sources(self) -> List[str]:
        if self.vector_store is None:
            self.load_vector_store()
        
        collection = self.vector_store._collection
        all_metadatas = collection.get()['metadatas']

        sources = set()
        for metadata in all_metadatas:
            if 'source' in metadata:
                sources.add(metadata['source'])
        
        return sorted(list(sources))
    
    def print_search_results(
            self,
            query: str,
            results: List[Dict[str, Any]],
            max_content_length: int = 300
    ) -> None:
        print("\n" + "="*80)
        print(f"SEARCH RESULTS FOR: '{query}'")
        print("="*80)

        if not results:
            print("\nNo results found.")
            return

        for result in results:
            print(f"\n[Rank {result['rank']}]")
            
            if 'similarity_score' in result:
                print(f"Similarity Score: {result['similarity_score']:.4f}")
            
            print(f"Source: {Path(result['source']).name}")
            print(f"Page: {result['page']}")
            
            content = result['content']
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            print(f"\nContent Preview:")
            print(f"{content}")
            print("-" * 80)

        print()


    def evaluate_retrieval(
            self,
            test_queries: List[str],
            expected_sources: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate retrieval performance using test queries."""
        logger.info("Starting retrieval evaluation...")
        logger.info(f"Number of test queries: {len(test_queries)}")

        results = {
            "total_queries": len(test_queries),
            "queries_with_results": 0,
            "average_score": 0.0,
            "average_results_per_query": 0.0,
            "query_results": []
        }

        total_score = 0.0
        total_results = 0

        for i, query in enumerate(test_queries):
            query_results = self.retrieve_documents(query, return_scores=True)

            if query_results:
                results["queries_with_results"] += 1
                total_results += len(query_results)

                # calculer le score moyen pour cette requête
                query_avg_score = sum([res['similarity_score'] for res in query_results]) / len(query_results)
                total_score += query_avg_score

                # verifier les sources attendues si fournies
                retrieved_sources = [Path(r["source"]).name for r in query_results]
                matches = None
                if expected_sources and i < len(expected_sources):
                    expected = expected_sources[i]
                    matches = len(set(retrieved_sources) & set(expected))

                results["query_results"].append({
                    "query": query,
                    "num_results": len(query_results),
                    "avg_score": query_avg_score,
                    "top_score": query_results[0]["similarity_score"],
                    "sources": retrieved_sources,
                    "expected_matches": matches
                })

        # Calculate overall metrics
        if results["queries_with_results"] > 0:
            results["average_score"] = total_score / results["queries_with_results"]
            results["average_results_per_query"] = total_results / len(test_queries)

        logger.info(f"Evaluation complete: {results['queries_with_results']}/{results['total_queries']} queries returned results")
        return results
    


# to test the retriever module
if __name__ == "__main__":
    # Initialize retriever
    retriever = Retriever(
        vector_store_path="./chroma_db",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="documents",
        top_k=5
    )

    # Load vector store
    try:
        retriever.load_vector_store()

        # Example queries for testing
        test_queries = [
            "What are the main findings of this research?",
            "Explain the methodology used in the study",
            "What are the conclusions and future work?"
        ]

        print("\nRunning test queries...\n")

        for query in test_queries:
            # Retrieve documents
            results = retriever.retrieve_documents(query, top_k=3)

            # Print results
            retriever.print_search_results(query, results)

        # Get unique sources
        sources = retriever.get_unique_sources()
        print(f"\nUnique sources in database: {len(sources)}")
        for source in sources:
            print(f"  - {Path(source).name}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise