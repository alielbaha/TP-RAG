import os
from typing import List, Optional, Dict 
from pathlib import Path
import logging 

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain_core.documents import Document


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
            if file_type.lower() == "pdf":
                loader = DirectoryLoader(data_path, glob="**/*.pdf" if recursive else "*.pdf", loader_cls = PyPDFLoader, show_progress =True)
                docs = loader.load()  
                logger.info(f"chargmet {len(docs)} pdfs")

            else :
                raise ValueError(f"pdfs chargés")

            return docs

    def split_docs(self, docs, preserve_metadata = True):
        logger.info(f"splitting {len(docs)} to chunks")

        text_splitter = self.init_text_splitter()
        chunks = text_splitter.split_documents(docs)

        if preserve_metadata:
            for i,chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(f"{len(chunks)} chunks crées a partir de {len(docs)} docs")

        return chunks


    def create_embeddings(self, chunks):
        self.initialize_embedding()
        logger.info(f"Embeddings crée pour {len(chunks)} chunks")
        return chunks 


    def store_in_vector_db(self, chunks, persist = True):
        logger.info(f"stockage {len(chunks)} chunks")
        embeddings = self._init_embedding()

        if persist:
            os.makedirs(self.vector_store_path, exist_ok = True)

        self.vector_store = Chroma.from_documents(docs = chunks,  embedding = embeddings, collection_name = self.collection_name, presist_directory = self.vector_store_path if persist else None)

        logger.info("vector store crée dans {self.vector_store_path}")
        return self.vector_store


    def index_doc(self, data_path,file_type = "pdf", persist = True):
        logger.info("demarrage d'indexation..")

        docs = self.load_doc(data_path, file_type)

        chunks = self.split_documents(docs)

        chunks = self.create_embeddings(chunks)

        vector_store = self.store_in_vector_db(chunks, persist)

        logger.info("Done")


        return vector_store

    def load_existing_vector_store(self):
        logger.info("chargemnt de vecore store dans {self.vector_store_path}")

        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError(f"vector store introuvable dans {self.vector_store_path}")
        
        embeddings = self._init_embedding()

        self.vector_store = Chroma(collection_name = self.collection_name, embedding_function = embeddings, persist_directory = self.vector_store_path)

        logger.info("vector store chargé")
        return self.vector_store
    
    def get_stats(self):
        if self.vector_store is None:
            return {"erreur" : "pas de vector store"}
        collection = self.vectore_store_collection 
        count = collection.count()


        stats = {"nb_chunks_total" : count,
                  "embedding_model" : self.embedding_model_name,
                  "collection_name" : self.collection_name,
                  "vector_store_path" : self.vector_store_path,
                  "chunk_size": self.chunk_size,
                  "chunk_overlap":self.chunk_overlap}
        

        logger.info(f"stats du vector store : {stats}")
        return stats
    



if __name__ == "__main__":
    indexer = DocIndexer(embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2")

    try:
        vector_store = indexer.index_doc(data_path = "./data", file_type = "pdf", persist = True)
        stats = indexer.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        logger.error(f"error during indexing: {e} ")
        raise


    



    
        
        
        
