#!/usr/bin/env python3
"""
Simplified RAG System using EmbeddingGemma + Ollama

This version uses:
- EmbeddingGemma-300M for creating high-quality embeddings
- Ollama for text generation
- Basic text extraction from PDFs (no vision or table processing)
"""

import os
import logging
from pathlib import Path
import argparse
from typing import List, Dict, Any

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from embedding import EmbeddingClass


class PDFRAG:
    """
    Simplified RAG system using EmbeddingGemma for embeddings and Ollama for generation.
    
    This version focuses on basic text extraction without vision or table processing.
    """
    
    def __init__(
        self, 
        pdf_folder: str = "datasets", 
        ollama_model: str = "gemma3:1b",
        embedding_model: str = "embeddinggemma:300m",
        persist_directory: str = "./chroma_db",
        k_similar_chunks: int = 2,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the simplified RAG system.
        
        Args:
            pdf_folder (str): Path to folder containing PDF files
            ollama_model (str): Ollama model for text generation
            embedding_model (str): Ollama embedding model
            persist_directory (str): Directory to persist ChromaDB data
            k_similar_chunks (int): Number of similar chunks to retrieve
            chunk_size (int): Maximum size of text chunks for embedding
            chunk_overlap (int): Overlap between consecutive text chunks
        """
        self.pdf_folder = Path(pdf_folder)
        self.ollama_model = ollama_model
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.k_similar_chunks = k_similar_chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        
        logger.info("🚀 Initializing Simplified RAG System")
        logger.info(f"📚 PDF folder: {self.pdf_folder}")
        logger.info(f"🤖 Ollama model: {ollama_model}")
        logger.info(f"🧠 Embedding model: {embedding_model}")
        logger.info(f"💾 Persist directory: {persist_directory}")
        logger.info(f"🔍 Chunking: {chunk_size} chars (overlap: {chunk_overlap})")
    
    def load_pdfs(self) -> List[Document]:
        """
        Load all PDF files from the specified folder and extract text.
        
        Returns:
            List[Document]: List of documents with text and metadata
        """
        logger.info(f"📚 Loading PDFs from {self.pdf_folder}")
        documents = []
        
        if not self.pdf_folder.exists():
            logger.error(f"PDF folder not found: {self.pdf_folder}")
            return documents
        
        pdf_files = list(self.pdf_folder.glob("**/*.pdf"))
        logger.info(f"📄 Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"📖 Processing: {pdf_file.name}")
                
                # Extract text using pypdf
                with open(pdf_file, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    text_content = []
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"--- Page {page_num} ---\n{page_text}")
                    
                    full_text = "\n\n".join(text_content)
                    
                    if full_text.strip():
                        doc = Document(
                            page_content=full_text,
                            metadata={
                                "source": str(pdf_file),
                                "filename": pdf_file.name,
                                "pages": len(pdf_reader.pages)
                            }
                        )
                        documents.append(doc)
                        logger.info(f"✅ Extracted {len(pdf_reader.pages)} pages from {pdf_file.name}")
                    else:
                        logger.warning(f"⚠️ No text found in {pdf_file.name}")
                        
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"🎉 Successfully loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents (List[Document]): List of documents to chunk
            
        Returns:
            List[Document]: List of chunked documents
        """
        logger.info("✂️ Chunking documents")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunked_docs = text_splitter.split_documents(documents)
        
        logger.info(f"📦 Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def setup_embeddings_and_llm(self):
        """
        Initialize EmbeddingGemma for embeddings and Ollama for text generation.
        """
        logger.info("🔧 Setting up embedding + generation system")
        
        try:
            # Initialize EmbeddingGemma via Ollama
            self.embeddings = EmbeddingClass(model_name=self.embedding_model)
            logger.info("✅ EmbeddingGemma embeddings initialized")
            
            # Initialize Ollama LLM for generation
            self.llm = OllamaLLM(model=self.ollama_model)
            logger.info(f"✅ Ollama LLM ({self.ollama_model}) initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
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
                persist_directory=self.persist_directory
            )
            logger.info("✅ Vector store created and persisted")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
    
    def load_existing_vectorstore(self) -> bool:
        """
        Load an existing ChromaDB vector store if it exists.
        
        Returns:
            bool: True if existing store loaded successfully
        """
        if os.path.exists(self.persist_directory):
            try:
                logger.info(f"📂 Loading existing vector store from {self.persist_directory}")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("✅ Existing vector store loaded")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing store: {str(e)}")
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
        
        # Prompt template
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
        Initialize the complete RAG system.
        
        Args:
            force_rebuild (bool): If True, rebuild vector store from scratch
        """
        logger.info("🚀 Initializing RAG System...")
        
        # Setup embedding + generation system
        self.setup_embeddings_and_llm()
        
        # Load existing or create new vector store
        if not force_rebuild and self.load_existing_vectorstore():
            logger.info("✅ Using existing vector store")
        else:
            logger.info("📥 Building new vector store from PDFs...")
            documents = self.load_pdfs()
            chunked_docs = self.chunk_documents(documents)
            self.create_vectorstore(chunked_docs)
        
        # Setup QA chain
        self.setup_qa_chain()
        
        logger.info("🎉 RAG system initialization complete!")
        logger.info("💡 Ready to answer questions about your PDF documents!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict[str, Any]: Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        logger.info(f"🔍 Processing query: {question[:50]}...")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": []
            }


def main():
    """
    Main function demonstrating the simplified RAG system.
    """
    parser = argparse.ArgumentParser(description="Simplified PDF RAG System")
    parser.add_argument("--pdf-folder", type=str, default="datasets", help="Path to PDF folder")
    parser.add_argument("--ollama-model", type=str, default="gemma3:1b", help="Ollama model for generation")
    parser.add_argument("--embedding-model", type=str, default="embeddinggemma:300m", help="Embedding model")
    parser.add_argument("--persist-directory", type=str, default="./chroma_db", help="Vector store directory")
    parser.add_argument("--k-similar-chunks", type=int, default=2, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild vector store")
    
    args = parser.parse_args()
    
    print("🧠 Simplified PDF RAG System")
    print("🤝 EmbeddingGemma-300M via Ollama")
    print("=" * 60)
    
    try:
        # Initialize RAG system
        rag_system = PDFRAG(
            pdf_folder=args.pdf_folder,
            ollama_model=args.ollama_model,
            embedding_model=args.embedding_model,
            persist_directory=args.persist_directory,
            k_similar_chunks=args.k_similar_chunks,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        print("\n⚙️ Initializing RAG system...")
        rag_system.initialize_system(force_rebuild=args.force_rebuild)
        
        print("\n🎉 RAG system ready!")
        print("💡 Type 'quit' or 'exit' to stop.\n")
        
        # Interactive query loop
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Get answer
                result = rag_system.query(question)
                
                print("\n" + "=" * 60)
                print("📝 Answer:")
                print(result["answer"])
                
                if result["source_documents"]:
                    print("\n📚 Sources:")
                    for i, doc in enumerate(result["source_documents"], 1):
                        source = doc.metadata.get("filename", "Unknown")
                        print(f"  {i}. {source}")
                
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        print("\n💡 Make sure:")
        print("  - Ollama is running (ollama serve)")
        print(f"  - Embedding model is available (ollama pull {args.embedding_model})")
        print(f"  - Ollama LLM model is available (ollama pull {args.ollama_model})")
        print(f"  - PDF files are in the {args.pdf_folder} folder")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
