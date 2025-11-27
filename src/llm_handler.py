# src/llm_handler.py


import logging
from typing import List, Tuple, Optional, Dict, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
import os

logger = logging.getLogger(__name__)


class LLMHandler:

    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        prompt_template: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        huggingface_api_token: str = "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"
    ):
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self.api_token = huggingface_api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        
        if not self.api_token:
            logger.warning(
                "Aucun token HuggingFace trouvé. "
                "Définissez HUGGINGFACE_API_TOKEN dans l'environnement ou passez-le en paramètre."
            )
        
        self.default_template = """Tu es un assistant IA expert qui répond aux questions en se basant strictement sur le contexte fourni.

Contexte:
{context}

Question: {question}

Instructions:
- Réponds de manière précise et concise
- Base-toi UNIQUEMENT sur les informations du contexte
- Si l'information n'est pas dans le contexte, dis clairement "Je ne trouve pas cette information dans les documents fournis"
- Cite les sources (numéros de document) quand c'est pertinent
- Utilise un ton professionnel mais accessible

Réponse:"""
        
        self.prompt_template = prompt_template or self.default_template
        
        self._init_llm()
        self._init_chain()
        
        logger.info(f"LLMHandler initialisé avec le modèle: {model_name}")
    
    def _init_llm(self):
        try:
            self.llm = HuggingFaceHub(
                repo_id=self.model_name,
                huggingfacehub_api_token=self.api_token,
                model_kwargs={
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "repetition_penalty": 1.1,
                    "return_full_text": False
                }
            )
            logger.info("LLM initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du LLM: {e}")
            raise
    
    def _init_chain(self):
        try:
            self.prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "question"]
            )
            
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                verbose=False
            )
            
            logger.info("Chaîne LLM créée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la création de la chaîne: {e}")
            raise
    
    def format_context(
        self,
        retrieved_docs: List[Tuple[Document, float]],
        max_context_length: int = 3000
    ) -> str:
        """
        Formate les documents récupérés en contexte pour le prompt.
        
        Args:
            retrieved_docs: Liste de tuples (Document, score)
            max_context_length: Longueur maximale du contexte en caractères
            
        Returns:
            Contexte formaté sous forme de chaîne
        """
        if not retrieved_docs:
            return "Aucun document pertinent trouvé."
        
        context_parts = []
        current_length = 0
        
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Source inconnue')
            page = doc.metadata.get('page', 'N/A')
            
            doc_text = f"\n--- Document {i} ---"
            doc_text += f"\nSource: {source}"
            doc_text += f"\nPage: {page}"
            doc_text += f"\nScore de pertinence: {score:.3f}"
            doc_text += f"\nContenu:\n{doc.page_content}\n"
            
            if current_length + len(doc_text) > max_context_length:
                logger.warning(
                    f"Limite de contexte atteinte. "
                    f"Utilisation de {i-1}/{len(retrieved_docs)} documents."
                )
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Tuple[Document, float]],
        return_metadata: bool = False
    ) -> str:
        """
        Génère une réponse à partir de la question et du contexte récupéré.
        
        Args:
            question: Question de l'utilisateur
            retrieved_docs: Documents récupérés avec scores
            return_metadata: Si True, retourne aussi les métadonnées
            
        Returns:
            Réponse générée par le LLM (et métadonnées si demandé)
        """
        if not question.strip():
            raise ValueError("La question ne peut pas être vide")
        
        logger.info(f"Génération de réponse pour: '{question[:50]}...'")
        
        context = self.format_context(retrieved_docs)
        
        if not retrieved_docs:
            default_response = (
                "Je suis désolé, mais je n'ai trouvé aucun document pertinent "
                "pour répondre à votre question. Pourriez-vous reformuler ou "
                "poser une question différente ?"
            )
            return default_response if not return_metadata else {
                "answer": default_response,
                "context": context,
                "sources_used": []
            }
        
        try:
            response = self.chain.run(
                context=context,
                question=question
            )
            
            answer = self._clean_response(response)
            
            logger.info("Réponse générée avec succès")
            
            if return_metadata:
                return {
                    "answer": answer,
                    "context": context,
                    "sources_used": [
                        {
                            "source": doc.metadata.get('source', 'N/A'),
                            "page": doc.metadata.get('page', 'N/A'),
                            "score": score
                        }
                        for doc, score in retrieved_docs
                    ],
                    "num_tokens_estimate": len(answer.split())
                }
            
            return answer
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            error_response = (
                "Une erreur s'est produite lors de la génération de la réponse. "
                "Veuillez réessayer."
            )
            return error_response if not return_metadata else {
                "answer": error_response,
                "error": str(e)
            }
    
    def _clean_response(self, response: str) -> str:
        """
        Nettoie la réponse générée par le LLM.
        
        Args:
            response: Réponse brute du LLM
            
        Returns:
            Réponse nettoyée
        """
        response = response.strip()
        
        if "Réponse:" in response:
            response = response.split("Réponse:")[-1].strip()
        
        response = response.replace("```", "").strip()
        
        return response
    
    def update_prompt_template(self, new_template: str):
        if "{context}" not in new_template or "{question}" not in new_template:
            raise ValueError(
                "Le template doit contenir les variables {context} et {question}"
            )
        
        self.prompt_template = new_template
        self._init_chain()
        logger.info("Template de prompt mis à jour")
    
    def batch_generate(
        self,
        questions: List[str],
        retrieved_docs_list: List[List[Tuple[Document, float]]]
    ) -> List[str]:

        if len(questions) != len(retrieved_docs_list):
            raise ValueError(
                "Le nombre de questions doit correspondre au nombre de listes de documents"
            )
        
        logger.info(f"Génération batch de {len(questions)} réponses")
        
        answers = []
        for question, docs in zip(questions, retrieved_docs_list):
            answer = self.generate_answer(question, docs)
            answers.append(answer)
        
        return answers
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle et la configuration.
        
        Returns:
            Dictionnaire avec les informations du modèle
        """
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "prompt_template_length": len(self.prompt_template),
            "has_api_token": self.api_token is not None
        }


if __name__ == "__main__":
    print("Test du LLM Handler...")
    from langchain.schema import Document
    test_doc = Document(
        page_content="Le machine learning est une branche de l'intelligence artificielle.",
        metadata={"source": "test.pdf", "page": 1}
    )
    try:
        handler = LLMHandler()
        answer = handler.generate_answer(
            question="Qu'est-ce que le machine learning?",
            retrieved_docs=[(test_doc, 0.95)]
        )
        print(f"\nRéponse générée: {answer}")   
    except Exception as e:
        print(f"Erreur: {e}")
        print("\nPour utiliser ce module, définissez HUGGINGFACE_API_TOKEN:")
        print("export HUGGINGFACE_API_TOKEN='your_token_here'")