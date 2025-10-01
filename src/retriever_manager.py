import chromadb
import os
import json
from typing import List, Optional, Union

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool


class RetrieverManager:
    def __init__(self, persist_directory: str, collection_name: str, embedding_model: str = "models/gemini-embedding-001"):
        """
        Initialize the RetrieverManager.
        
        Args:
            persist_directory: Directory path for persisting the vector store
            collection_name: Name of the ChromaDB collection
            embedding_model: Google Generative AI embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        self.retriever = None
        
        # Load text splitter configuration from config.json
        self.text_splitter_config = self._load_text_splitter_config()
        

    def _load_text_splitter_config(self):
        """
        Load text splitter configuration from config.json.
        
        Returns:
            dict: Text splitter configuration with chunk_size and chunk_overlap
        """
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
            return config.get("text_splitter", {})
        except (FileNotFoundError, json.JSONDecodeError):
            # Default values if config file is not available
            return {"chunk_size": 1000, "chunk_overlap": 200}
        

    def check_collection_exists(self) -> bool:
        """
        Check if a ChromaDB collection already exists.
        
        Returns:
            bool: True if the collection exists, False otherwise
        """
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            existing_collections = client.list_collections()
            return any(collection.name == self.collection_name for collection in existing_collections)
        except Exception as e:
            print(f"[ERROR] Failed to check collections: {e}")
            return False
    

    def create_collection_from_pdf(self, pdf_path: str) -> bool:
        """
        Create a new collection from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            bool: True if collection creation was successful, False otherwise
            
        Raises:
            FileNotFoundError: If the PDF file does not exist
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            pdf_loader = PyPDFLoader(pdf_path)
            pages = pdf_loader.load()
            print(f"[STATUS] PDF has been loaded, total {len(pages)} pages.")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.text_splitter_config.get("chunk_size", 1000),
                chunk_overlap=self.text_splitter_config.get("chunk_overlap", 200)
            )
            
            page_split = text_splitter.split_documents(pages)
            
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory)
            
            self.vectorstore = Chroma.from_documents(
                documents=page_split,
                embedding=self.embeddings, 
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            print(f"[STATUS] New ChromaDB collection '{self.collection_name}' has been created from PDF.")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to create collection from PDF: {e}")
            return False
    
    
    def create_collection_from_obsidian(self, obsidian_path: str) -> bool:
        """
        Create a new collection from an Obsidian vault.
        
        Args:
            obsidian_path: Path to the Obsidian vault directory
            
        Returns:
            bool: True if collection creation was successful, False otherwise
            
        Raises:
            FileNotFoundError: If the Obsidian vault directory does not exist
        """
        if not os.path.exists(obsidian_path):
            raise FileNotFoundError(f"Obsidian vault not found: {obsidian_path}")
        
        try:
            # Load all markdown files from the Obsidian vault using TextLoader
            loader = DirectoryLoader(
                obsidian_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                show_progress=True,
                use_multithreading=True,
                loader_kwargs={'encoding': 'utf-8'}
            )
            
            documents = loader.load()
            print(f"[STATUS] Obsidian vault loaded, total {len(documents)} markdown files.")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.text_splitter_config.get("chunk_size", 1000),
                chunk_overlap=self.text_splitter_config.get("chunk_overlap", 200)
            )
            
            split_documents = text_splitter.split_documents(documents)
            
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory)
            
            self.vectorstore = Chroma.from_documents(
                documents=split_documents,
                embedding=self.embeddings, 
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            print(f"[STATUS] New ChromaDB collection '{self.collection_name}' has been created from Obsidian vault.")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to create collection from Obsidian vault: {e}")
            return False
    

    def load_existing_collection(self) -> bool:
        """
        Load an existing ChromaDB collection.
        
        Returns:
            bool: True if collection was loaded successfully, False otherwise
        """
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print(f"[STATUS] Existing ChromaDB collection '{self.collection_name}' loaded successfully.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load existing collection: {e}")
            return False
    

    def setup_retriever(self, search_type: str = "similarity", k: int = 5):
        """
        Setup the retriever with given parameters.
        
        Args:
            search_type: Type of search to perform ("similarity", "mmr", etc.)
            k: Number of documents to retrieve
            
        Raises:
            RuntimeError: If vector store is not initialized
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized")
        
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        print(f"[STATUS] Retriever initialized with search_type='{search_type}' and k={k}")
    

    def initialize_retriever(self, pdf_path: Optional[str] = None, obsidian_path: Optional[str] = None, search_type: str = "similarity", k: int = 5):
        """
        Initialize the retriever, creating collection from either PDF or Obsidian vault if needed.
        
        Args:
            pdf_path: Optional path to PDF file for creating new collection
            obsidian_path: Optional path to Obsidian vault for creating new collection
            search_type: Type of search to perform
            k: Number of documents to retrieve
            
        Returns:
            Retriever: Initialized retriever instance
            
        Raises:
            ValueError: If both PDF and Obsidian paths are specified
            RuntimeError: If collection loading/creation fails
        """
        collection_exists = self.check_collection_exists()
        
        if collection_exists:
            print(f"[STATUS] Collection '{self.collection_name}' exists. Loading...")
            if self.load_existing_collection():
                self.setup_retriever(search_type, k)
            else:
                raise RuntimeError("Failed to load existing collection")
        else:
            print(f"[STATUS] Collection '{self.collection_name}' not found. Creating new collection...")
            
            if pdf_path and obsidian_path:
                raise ValueError("Cannot specify both PDF path and Obsidian path. Choose one source.")
            
            if pdf_path:
                if self.create_collection_from_pdf(pdf_path):
                    self.setup_retriever(search_type, k)
                else:
                    raise RuntimeError("Failed to create collection from PDF")
            elif obsidian_path:
                if self.create_collection_from_obsidian(obsidian_path):
                    self.setup_retriever(search_type, k)
                else:
                    raise RuntimeError("Failed to create collection from Obsidian vault")
            else:
                raise RuntimeError("Either PDF path or Obsidian path is required to create new collection")
        
        return self.retriever
    

    def get_retriever_tool(self):
        """
        Create and return the retriever tool.
        
        Returns:
            Tool: LangChain tool for document retrieval
            
        Raises:
            RuntimeError: If retriever is not initialized
        """
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Call initialize_retriever() first.")
        
        
        @tool
        def retriever_tool(query: str) -> str:
            """
            Search and return information from the database.
            
            Args:
                query: Search query to find relevant documents
                
            Returns:
                str: Retrieved documents concatenated as a string
            """
            docs = self.retriever.invoke(query)
            
            if not docs:
                return "No relevant information found in the database"
            
            results = []
            for i, doc in enumerate(docs):
                results.append(f"Document {i+1}:\n{doc.page_content}")
            
            return "\n\n".join(results)
        
        return retriever_tool