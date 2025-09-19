#!/usr/bin/env python3
"""
RAG System using EmbeddingGemma + Ollama

This version uses:
- EmbeddingGemma-300M for creating high-quality embeddings (specialized for this task)
- Ollama for text generation (what it's good at)
- Task-specific prompts optimized for retrieval

"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, continue without it

import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingGemmaEmbeddings(Embeddings):
    """
    Custom embeddings class using EmbeddingGemma-300M via SentenceTransformers.
    
    This class implements task-specific prompts for optimal retrieval performance:
    - Query prompts: "task: search result | query: {text}"
    - Document prompts: "title: none | text: {text}"
    """
    
    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        """
        Initialize EmbeddingGemma embeddings.
        
        Args:
            model_name (str): HuggingFace model name for EmbeddingGemma
        """
        self.model_name = model_name
        logger.info(f"Loading EmbeddingGemma model: {model_name}")
        
        try:
            # Check HuggingFace authentication
            self._check_hf_auth()
            
            # Load the model using SentenceTransformers
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            logger.info("✅ EmbeddingGemma model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load EmbeddingGemma: {str(e)}")
            self._show_auth_help()
            raise
    
    def _check_hf_auth(self):
        """Check if HuggingFace authentication is set up."""
        try:
            from huggingface_hub import whoami
            user_info = whoami()
            logger.info(f"🔐 Authenticated as: {user_info.get('name', 'Unknown')}")
        except Exception:
            logger.warning("⚠️ Not authenticated with HuggingFace")
            self._show_auth_help()
            
    def _show_auth_help(self):
        """Show authentication help message."""
        logger.error("🔑 HuggingFace Authentication Required!")
        logger.error("To use EmbeddingGemma, you need to:")
        logger.error("1. Accept license: https://huggingface.co/google/embeddinggemma-300m")
        logger.error("2. Authenticate using one of these methods:")
        logger.error("   a) Run: python set_hf_token.py")
        logger.error("   b) Set HUGGINGFACE_HUB_TOKEN environment variable")
        logger.error("   c) Use: huggingface-cli login (if available)")
        logger.error("3. Get token from: https://huggingface.co/settings/tokens")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using the document-specific prompt format.
        
        According to EmbeddingGemma documentation, document embeddings should use:
        "title: {title | 'none'} | text: {content}"
        
        Args:
            texts (List[str]): List of document texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        # Format texts with document prompt
        formatted_texts = [f"title: none | text: {text}" for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(formatted_texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query using the query-specific prompt format.
        
        According to EmbeddingGemma documentation, query embeddings should use:
        "task: search result | query: {content}"
        
        Args:
            text (str): Query text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        # Format text with query prompt
        formatted_text = f"task: search result | query: {text}"
        
        # Generate embedding
        embedding = self.model.encode([formatted_text], convert_to_tensor=False)
        return embedding[0].tolist()


class ImprovedPDFRAG:
    """
    RAG system using EmbeddingGemma for embeddings and Ollama for generation.
    
    This hybrid approach provides:
    - Superior embedding quality with EmbeddingGemma-300M
    - Efficient text generation with Ollama
    - Task-specific prompts for optimal retrieval
    - Better performance and resource usage
    """
    
    def __init__(self, pdf_folder: str = "datasets", 
                 ollama_model: str = "gemma3:1b",
                 embedding_model: str = "google/embeddinggemma-300m",
                 persist_directory: str = "./chroma_db",
                 k_similar_chunks: int = 2):
        """
        Initialize the RAG system.
        
        Args:
            pdf_folder (str): Path to folder containing PDF files
            ollama_model (str): Ollama model for text generation
            embedding_model (str): HuggingFace embedding model
            persist_directory (str): Directory to persist ChromaDB data
            k_similar_chunks (int): Number of similar chunks to retrieve
        """
        self.pdf_folder = Path(pdf_folder)
        self.ollama_model = ollama_model
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.k_similar_chunks = k_similar_chunks
        # Initialize components
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        
        logger.info("🚀 Initializing  RAG System")
        logger.info(f"📚 PDF folder: {self.pdf_folder}")
        logger.info(f"🤖 Ollama model: {ollama_model}")
        logger.info(f"🧠 Embedding model: {embedding_model}")
        logger.info(f"💾 Persist directory: {persist_directory}")
    
    def load_pdfs(self) -> List[Document]:
        """
        Load all PDF files from the specified folder and extract text.
        
        Same implementation as the original system.
        
        Returns:
            List[Document]: List of documents with text and metadata
        """
        logger.info(f"📚 Loading PDFs from {self.pdf_folder}")
        documents = []
        
        if not self.pdf_folder.exists():
            logger.error(f"❌ PDF folder {self.pdf_folder} does not exist!")
            return documents
        
        pdf_files = list(self.pdf_folder.glob("**/*.pdf"))
        logger.info(f"📄 Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"🔄 Processing: {pdf_file.name}")
                
                with open(pdf_file, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    text_content = ""
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                if text_content.strip():
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "source": str(pdf_file),
                            "filename": pdf_file.name,
                            "total_pages": len(pdf_reader.pages)
                        }
                    )
                    documents.append(doc)
                    logger.info(f"✅ Loaded {pdf_file.name} ({len(pdf_reader.pages)} pages)")
                else:
                    logger.warning(f"⚠️ No text content found in {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_file.name}: {str(e)}")
        
        logger.info(f"🎉 Successfully loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Uses the same chunking strategy as the original system.
        
        Args:
            documents (List[Document]): List of documents to chunk
            
        Returns:
            List[Document]: List of chunked documents
        """
        logger.info("✂️ Chunking documents for optimal retrieval")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunked_docs = text_splitter.split_documents(documents)
        
        logger.info(f"📦 Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def setup_embeddings_and_llm(self):
        """
        Initialize EmbeddingGemma for embeddings and Ollama for text generation.
        
        This hybrid approach provides:
        - Better embedding quality with specialized EmbeddingGemma
        - Efficient text generation with Ollama
        """
        logger.info("🔧 Setting up hybrid embedding + generation system")
        
        try:
            # Initialize EmbeddingGemma for embeddings
            logger.info("🧠 Loading EmbeddingGemma for embeddings...")
            self.embeddings = EmbeddingGemmaEmbeddings(model_name=self.embedding_model)
            
            # Initialize Ollama for text generation
            logger.info(f"🤖 Connecting to Ollama for generation ({self.ollama_model})...")
            self.llm = OllamaLLM(
                model=self.ollama_model,
                base_url="http://localhost:11434",
                temperature=0.1
            )
            
            logger.info("✅ Hybrid system initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing hybrid system: {str(e)}")
            logger.error("Make sure:")
            logger.error("  - Internet connection for downloading EmbeddingGemma")
            logger.error("  - Ollama is running (ollama serve)")
            logger.error(f"  - Model available (ollama pull {self.ollama_model})")
            raise
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """
        Create ChromaDB vector store using EmbeddingGemma embeddings.
        
        Args:
            documents (List[Document]): Chunked documents to store
        """
        logger.info("🗄️ Creating vector store with EmbeddingGemma embeddings")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="pdf_rag_embeddinggemma"
            )
            
            logger.info(f"✅ Vector store created with {len(documents)} chunks")
            logger.info(f"💾 Persisted to: {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {str(e)}")
            raise
    
    def load_existing_vectorstore(self) -> bool:
        """
        Load an existing ChromaDB vector store if it exists.
        
        Returns:
            bool: True if existing store loaded successfully
        """
        if os.path.exists(self.persist_directory):
            try:
                logger.info("📂 Loading existing vector store...")
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name="pdf_rag_embeddinggemma"
                )
                logger.info("✅ Existing vector store loaded successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Error loading existing vector store: {str(e)}")
                return False
        return False
    
    def setup_qa_chain(self):
        """
        Set up the question-answering chain with optimized prompts.
        """
        logger.info("🔗 Setting up QA chain")
        
        # Create retriever with similarity search
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k_similar_chunks}
        )
        
        # Enhanced prompt template
        prompt_template = """You are a helpful research assistant that provides accurate answers based on the provided PDF documents.

Use the following context from the documents to answer the question. Be precise and cite information when possible.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the information isn't in the context, say "I don't have enough information in the provided documents"
- Be specific and mention relevant details from the documents
- If multiple documents discuss the topic, synthesize the information

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        logger.info(f"PROMPT: {PROMPT}")
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        logger.info("✅ QA chain setup complete")
    
    def initialize_system(self, force_rebuild: bool = False):
        """
        Initialize the complete  RAG system.
        
        Args:
            force_rebuild (bool): If True, rebuild vector store from scratch
        """
        logger.info("🚀 Initializing  RAG System...")
        
        # Setup hybrid embedding + generation system
        self.setup_embeddings_and_llm()
        
        # Load existing or create new vector store
        if not force_rebuild and self.load_existing_vectorstore():
            logger.info("♻️ Using existing vector store")
        else:
            logger.info("🆕 Creating new vector store from PDFs")
            documents = self.load_pdfs()
            if not documents:
                raise ValueError("❌ No documents loaded! Check PDF folder and files.")
            
            chunked_docs = self.chunk_documents(documents)
            self.create_vectorstore(chunked_docs)
        
        # Setup QA chain
        self.setup_qa_chain()
        
        logger.info("🎉  RAG system initialization complete!")
        logger.info("💡 Ready to answer questions about your PDF documents!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the  RAG system with a question.
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict[str, Any]: Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("❌ RAG system not initialized! Call initialize_system() first.")
        
        logger.info(f"🔍 Processing query: {question[:50]}...")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": []
            }
            
            # Add source information
            for doc in result.get("source_documents", []):
                source_info = {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                response["source_documents"].append(source_info)
            
            logger.info("✅ Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}")
            raise


def main():
    """
    Main function demonstrating the RAG system.
    """
    print("🧠 PDF RAG System")
    print("🤝 EmbeddingGemma-300M + Ollama")
    print("=" * 60)
    
    try:
        # Initialize RAG system
        rag_system = ImprovedPDFRAG(
            pdf_folder="datasets",
            ollama_model="gemma3:4b-it-qat",
            embedding_model="google/embeddinggemma-300m"
        )
        
        print("\n⚙️ Initializing RAG system...")
        print("📥 This will download EmbeddingGemma-300M on first run...")
        rag_system.initialize_system()
        
        print("\n🎉 RAG system ready!")
        print("💡 Now using EmbeddingGemma for embeddings + Ollama for generation")
        print("💡 Type 'quit' or 'exit' to stop.\n")
        
        # Interactive query loop
        while True:
            try:
                question = input("🔍 Ask a question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🤔 Thinking...")
                response = rag_system.query(question)
                
                print(f"\n📝 Answer:")
                print(response["answer"])
                
                if response["source_documents"]:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(response["source_documents"], 1):
                        print(f"  {i}. {source['filename']}")
                        print(f"     Preview: {source['content_preview']}")
                
                print("\n" + "─" * 60)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                logger.error(f"Query error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        print("\n💡 Make sure:")
        print("  - Internet connection (for downloading EmbeddingGemma)")
        print("  - Ollama is running (ollama serve)")
        print(f"  - Ollama model is available (ollama pull gemma3:1b)")
        print("  - PDF files are in the datasets folder")


if __name__ == "__main__":
    main()
