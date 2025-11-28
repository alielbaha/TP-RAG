
import logging
from typing import List, Dict, Any, Optional
import warnings

from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import Document
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig
)
import torch

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMQuestionAnswering:
    """
    A class to handle LLM-based question answering using retrieved context.
    Supports both local models and HuggingFace API.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        use_api: bool = False,
        api_token: Optional[str] = None,
        use_quantization: bool = False
    ):
        """
        Initialize the LLM Question Answering system.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cpu' or 'cuda')
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
            use_api: Whether to use HuggingFace API instead of local model
            api_token: HuggingFace API token (required if use_api=True)
            use_quantization: Whether to use 8-bit quantization (requires GPU)
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.use_api = use_api
        self.api_token = api_token
        self.use_quantization = use_quantization and device == "cuda"

        # Initialize components
        self.llm = None
        self.tokenizer = None

        logger.info(f"LLMQuestionAnswering initialized with model: {model_name}")
        logger.info(f"Device: {device}, API mode: {use_api}")

    def _load_local_model(self) -> HuggingFacePipeline:
        """
        Load a local HuggingFace model.

        Returns:
            HuggingFacePipeline instance
        """
        logger.info(f"Loading local model: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Determine model type
            model_type = self._get_model_type()

            # Setup quantization config if enabled
            quantization_config = None
            if self.use_quantization:
                logger.info("Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )

            # Load model based on type
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            else:  # causal LM
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )

            # Move to device if not using quantization
            if not self.use_quantization and self.device == "cuda":
                model = model.to(self.device)

            logger.info("Model loaded successfully")

            # Create pipeline
            task = "text2text-generation" if model_type == "seq2seq" else "text-generation"
            
            pipe = pipeline(
                task=task,
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                repetition_penalty=1.1
            )

            # Wrap in LangChain
            llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Pipeline created successfully")

            return llm

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_api_model(self) -> HuggingFaceHub:
        """
        Load model via HuggingFace API.

        Returns:
            HuggingFaceHub instance
        """
        logger.info(f"Using HuggingFace API for model: {self.model_name}")

        if not self.api_token:
            raise ValueError("API token required when use_api=True")

        llm = HuggingFaceHub(
            repo_id=self.model_name,
            huggingfacehub_api_token=self.api_token,
            model_kwargs={
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample
            }
        )

        logger.info("API model initialized")
        return llm

    def _get_model_type(self) -> str:
        """
        Determine if model is seq2seq or causal LM.

        Returns:
            "seq2seq" or "causal"
        """
        seq2seq_models = ["t5", "flan", "bart"]
        model_lower = self.model_name.lower()
        
        for model_type in seq2seq_models:
            if model_type in model_lower:
                return "seq2seq"
        
        return "causal"

    def load_model(self):
        """
        Load the LLM (either local or API).
        """
        if self.llm is None:
            if self.use_api:
                self.llm = self._load_api_model()
            else:
                self.llm = self._load_local_model()

    def create_prompt_template(
        self,
        template: Optional[str] = None,
        include_sources: bool = True
    ) -> PromptTemplate:
        """
        Create a prompt template for question answering.

        Args:
            template: Custom template string (optional)
            include_sources: Whether to include source citations

        Returns:
            PromptTemplate instance
        """
        if template is None:
            # Default template optimized for RAG
            if include_sources:
                template = """You are a helpful AI assistant that answers questions based on the provided context. 
Use ONLY the information from the context below to answer the question. 
If the answer cannot be found in the context, say "I don't have enough information to answer this question based on the provided context."
Be concise and accurate. If relevant, mention the source documents.

Context:
{context}

Question: {question}

Answer: Let me answer based on the provided context."""
            else:
                template = """You are a helpful AI assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the answer cannot be found in the context, say "I don't have enough information to answer this question based on the provided context."
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        logger.info("Prompt template created")
        return prompt

    def answer_question(
        self,
        question: str,
        context: str,
        prompt_template: Optional[PromptTemplate] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using the provided context.

        Args:
            question: User question
            context: Retrieved context from vector database
            prompt_template: Custom prompt template (optional)

        Returns:
            Dictionary containing answer and metadata
        """
        if self.llm is None:
            self.load_model()

        # Create prompt template if not provided
        if prompt_template is None:
            prompt_template = self.create_prompt_template()

        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)

        logger.info(f"Generating answer for: '{question}'")

        try:
            # Generate answer
            response = chain.invoke({
                "context": context,
                "question": question
            })

            # Extract answer from response
            answer = response.get("text", "").strip()

            # Post-process answer
            answer = self._post_process_answer(answer)

            result = {
                "question": question,
                "answer": answer,
                "context_length": len(context),
                "model": self.model_name
            }

            logger.info("Answer generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    def answer_with_retrieval(
        self,
        question: str,
        retriever,
        prompt_template: Optional[PromptTemplate] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a question by retrieving context and generating answer.

        Args:
            question: User question
            retriever: DocumentRetriever instance
            prompt_template: Custom prompt template (optional)
            top_k: Number of documents to retrieve

        Returns:
            Dictionary containing answer, sources, and metadata
        """
        logger.info(f"Answering question with retrieval: '{question}'")

        # Retrieve relevant context
        results = retriever.retrieve_documents(question, top_k=top_k)
        
        if not results:
            return {
                "question": question,
                "answer": "I couldn't find any relevant information to answer this question.",
                "sources": [],
                "context_length": 0,
                "model": self.model_name
            }

        # Get context
        context = retriever.get_relevant_context(question, top_k=top_k)

        # Generate answer
        answer_result = self.answer_question(question, context, prompt_template)

        # Add sources
        sources = []
        for result in results:
            sources.append({
                "source": result["source"],
                "page": result["page"],
                "score": result.get("similarity_score", 0.0),
                "content_preview": result["content"][:200]
            })

        answer_result["sources"] = sources
        answer_result["num_sources"] = len(sources)

        return answer_result

    def create_retrieval_qa_chain(
        self,
        vector_store,
        prompt_template: Optional[PromptTemplate] = None,
        chain_type: str = "stuff"
    ) -> RetrievalQA:
        """
        Create a LangChain RetrievalQA chain.

        Args:
            vector_store: Vector store instance
            prompt_template: Custom prompt template (optional)
            chain_type: Type of chain ("stuff", "map_reduce", "refine")

        Returns:
            RetrievalQA chain
        """
        if self.llm is None:
            self.load_model()

        if prompt_template is None:
            prompt_template = self.create_prompt_template()

        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

        logger.info(f"RetrievalQA chain created with chain_type={chain_type}")
        return qa_chain

    def _post_process_answer(self, answer: str) -> str:
        """
        Post-process the generated answer.

        Args:
            answer: Raw answer from model

        Returns:
            Cleaned answer
        """
        # Remove common artifacts
        answer = answer.strip()
        
        # Remove repeated newlines
        while "\n\n\n" in answer:
            answer = answer.replace("\n\n\n", "\n\n")

        # Handle common model artifacts
        prefixes_to_remove = [
            "Answer: ",
            "A: ",
            "Based on the context, ",
            "According to the context, "
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        return answer

    def batch_answer(
        self,
        questions: List[str],
        retriever,
        top_k: int = 5,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch.

        Args:
            questions: List of questions
            retriever: DocumentRetriever instance
            top_k: Number of documents to retrieve per question
            show_progress: Whether to show progress

        Returns:
            List of answer dictionaries
        """
        logger.info(f"Answering {len(questions)} questions in batch")

        results = []
        for i, question in enumerate(questions, 1):
            if show_progress:
                print(f"Processing question {i}/{len(questions)}...", end="\r")

            result = self.answer_with_retrieval(question, retriever, top_k=top_k)
            results.append(result)

        if show_progress:
            print(f"Completed {len(questions)} questions.            ")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "use_api": self.use_api,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "model_loaded": self.llm is not None
        }

        if self.tokenizer is not None:
            info["vocab_size"] = self.tokenizer.vocab_size

        return info


# Example usage
if __name__ == "__main__":
    from retriever import DocumentRetriever

    # Initialize LLM handler (using smaller model for demo)
    llm_handler = LLMQuestionAnswering(
        model_name="google/flan-t5-base",  # Small, fast model
        device="cpu",
        max_new_tokens=256,
        temperature=0.7
    )

    # Load model
    llm_handler.load_model()

    # Example context
    context = """
    Artificial Intelligence (AI) refers to the simulation of human intelligence 
    in machines. Machine Learning is a subset of AI that enables systems to learn 
    and improve from experience. Deep Learning is a subset of Machine Learning 
    that uses neural networks with multiple layers.
    """

    # Answer question
    question = "What is the relationship between AI, ML, and Deep Learning?"
    
    result = llm_handler.answer_question(question, context)
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Model: {result['model']}")
