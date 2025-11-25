import os
from typing import List, Optional, Dict 
from pathlib import Path
import logging 

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain.schema import Document


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocIndexer:

    def __init__(self, embedding_model_name, vector_store_path = "./chroma_db", collection_name = "documents", chunk_size = 512, chunk_overlap = 100, use_markdown_splitter = False):

        self.embedding_model_name = embedding_model_name
        self.vector_store_path = vector_store_path 
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap 
        self.use_markdown_splitter = use_markdown_splitter

        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None 

        logger.info(f"lancement de {embedding_model_name}")

    
    def init_embedding(self):

        if self.embeddings is None:
            logger.info(f"lancement d'embedding model {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name = self.embedding_model_name, model_kwargs = {'device': 'cuda'}, encode_kwargs = {'normalize_embedding':True})

            #logger.info(
            
        return self.embeddings
    
    def init_text_splitter(self):

        if self.text_splitter is None:
            if self.use_mardown_splitter:
                logger.info("on utilise Markdowntextsplitter pour le chunking")
                self.text_splitter = MarkdownTextSplitter(chunk_size = self.chunk_size, cunk_overlap = self.chunk_overlap)

            else:
                logger.info("on utilise RecursiveCharacterTextSplitter pour le chunking")
                self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap, length_function = len,seperators = ["\n\n", "\n", ". ", " ", ""], add_start_index = True)

        return self.text_splitter
    
    def load_doc(self, data_path,file_type = "pdf", recursive = True):
        logger.info(f"chargment de docs")
        if not os.path.exists(data_path):
            raise FileNotFoundError("fichier introuvable : {data_path}")
        
        docs = []
        if os.path.isfile(data_path):
            if os.path.endswith('.pdf'):
                loader = PyPDFLoader(data_path)
                docs = loader.load()
                logger.info(f"chargemt d'un pdf : {data_path}")
            
            else:
                raise ValueError(f"erreur de type de fichier: {file_type}")
        else:

        
        
        