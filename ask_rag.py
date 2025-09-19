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
import re
from pathlib import Path
from typing import List, Dict, Any

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

import pypdf
import pdfplumber
import fitz  
import requests
import base64
import io
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from input_arguments import parse_arguments

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


class PDFRAG:
    """
    RAG system using EmbeddingGemma for embeddings and Ollama for generation.
    
    This hybrid approach provides:
    - Superior embedding quality with EmbeddingGemma-300M
    - Efficient text generation with Ollama
    - Task-specific prompts for optimal retrieval
    - Better performance and resource usage
    """
    
    def __init__(
        self, pdf_folder: str = "datasets", 
                 ollama_model: str = "gemma3:1b",
                 embedding_model: str = "google/embeddinggemma-300m",
                 persist_directory: str = "./chroma_db",
                 k_similar_chunks: int = 2,
                 enable_vision: bool = True,
                 vision_model: str = "gemma3:4b",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_image_width: int = 50,
                 min_image_height: int = 50,
                 vision_timeout: int = 15,
                 max_vision_retries: int = 1,
                 max_failures_before_disable: int = 5
        ):
        """
        Initialize the RAG system.
        
        Args:
            pdf_folder (str): Path to folder containing PDF files
            ollama_model (str): Ollama model for text generation
            embedding_model (str): HuggingFace embedding model
            persist_directory (str): Directory to persist ChromaDB data
            k_similar_chunks (int): Number of similar chunks to retrieve
            enable_vision (bool): Enable vision processing for images
            vision_model (str): Ollama vision model for image analysis
            chunk_size (int): Maximum size of text chunks for embedding
            chunk_overlap (int): Overlap between consecutive text chunks
            min_image_width (int): Minimum image width in pixels
            min_image_height (int): Minimum image height in pixels
            vision_timeout (int): Timeout in seconds for vision model analysis
            max_vision_retries (int): Maximum retries for failed vision analysis
            max_failures_before_disable (int): Disable vision after this many failures
        """
        self.pdf_folder = Path(pdf_folder)
        self.ollama_model = ollama_model
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.k_similar_chunks = k_similar_chunks
        
        # Store chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        
        logger.info("🚀 Initializing Enhanced RAG System with Vision")
        logger.info(f"📚 PDF folder: {self.pdf_folder}")
        logger.info(f"🤖 Ollama model: {ollama_model}")
        logger.info(f"🧠 Embedding model: {embedding_model}")
        logger.info(f"💾 Persist directory: {persist_directory}")
        logger.info(f"🔍 Chunking: {chunk_size} chars (overlap: {chunk_overlap})")
        
        # Vision processing settings
        self.vision_model = vision_model
        self.process_images = enable_vision  # Use parameter to control vision processing
        self.min_image_size = (min_image_width, min_image_height)  # Skip very small images
        self.vision_timeout = vision_timeout  # Timeout to avoid hanging
        self.max_vision_retries = max_vision_retries  # Retries for failed vision analysis
        self.vision_failures = 0  # Track vision failures
        self.max_failures_before_disable = max_failures_before_disable  # Disable vision after too many failures
    
    def _extract_page_numbers(self, content: str) -> List[int]:
        """
        Extract page numbers from chunk content based on page markers.
        
        Args:
            content (str): Document chunk content
            
        Returns:
            List[int]: List of page numbers found in the content
        """
        # Find all page markers like "--- Page 5 ---"
        page_pattern = r'--- Page (\d+) ---'
        matches = re.findall(page_pattern, content)
        
        # Convert to integers and remove duplicates while preserving order
        page_numbers = []
        seen = set()
        for match in matches:
            page_num = int(match)
            if page_num not in seen:
                page_numbers.append(page_num)
                seen.add(page_num)
        
        return sorted(page_numbers)
    
    def _format_page_reference(self, page_numbers: List[int]) -> str:
        """
        Format page numbers into a readable reference string.
        
        Args:
            page_numbers (List[int]): List of page numbers
            
        Returns:
            str: Formatted page reference (e.g., "p. 5", "pp. 3-5", "pp. 2, 4-6")
        """
        if not page_numbers:
            return ""
        
        if len(page_numbers) == 1:
            return f"p. {page_numbers[0]}"
        
        # Group consecutive pages into ranges
        ranges = []
        start = page_numbers[0]
        end = start
        
        for i in range(1, len(page_numbers)):
            if page_numbers[i] == end + 1:
                end = page_numbers[i]
            else:
                # End of current range
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = page_numbers[i]
        
        # Add the final range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return f"pp. {', '.join(ranges)}"
    
    def _extract_image_references(self, content: str) -> List[str]:
        """
        Extract image references from chunk content.
        
        Args:
            content (str): Document chunk content
            
        Returns:
            List[str]: List of image references found in the content
        """
        import re
        
        # Find image markers like "[IMAGE 1 on Page 3]"
        image_pattern = r'\[IMAGE (\d+) on Page (\d+)\]'
        matches = re.findall(image_pattern, content)
        
        # Format as readable references
        image_refs = []
        for img_num, page_num in matches:
            image_refs.append(f"Image {img_num} (p. {page_num})")
        
        return image_refs
    
    def _format_table_as_text(self, table: List[List[str]], table_index: int) -> str:
        """
        Format a table extracted by pdfplumber as readable text for RAG processing.
        
        Args:
            table (List[List[str]]): Table data from pdfplumber
            table_index (int): Index of the table on the page
            
        Returns:
            str: Formatted table text
        """
        if not table or not any(table):
            return ""
        
        try:
            # Filter out None values and convert to strings
            cleaned_table = []
            for row in table:
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cleaned_row):  # Skip completely empty rows
                    cleaned_table.append(cleaned_row)
            
            if not cleaned_table:
                return ""
            
            # Calculate column widths for formatting
            max_widths = []
            for col_idx in range(len(cleaned_table[0])):
                max_width = max(len(row[col_idx]) if col_idx < len(row) else 0 
                               for row in cleaned_table)
                max_widths.append(min(max_width, 30))  # Cap at 30 chars per column
            
            # Format table with proper spacing
            formatted_lines = [f"\n[TABLE {table_index}]"]
            
            for row_idx, row in enumerate(cleaned_table):
                formatted_row = " | ".join(
                    cell[:max_widths[col_idx]].ljust(max_widths[col_idx])
                    for col_idx, cell in enumerate(row[:len(max_widths)])
                )
                formatted_lines.append(formatted_row)
                
                # Add separator line after header (first row)
                if row_idx == 0:
                    separator = "-+-".join("-" * width for width in max_widths)
                    formatted_lines.append(separator)
            
            formatted_lines.append("")  # Empty line after table
            return "\n".join(formatted_lines)
            
        except Exception as e:
            logger.warning(f"⚠️ Error formatting table {table_index}: {str(e)}")
            return f"\n[TABLE {table_index}] - Error formatting table content\n"
    
    def _analyze_image_with_vision_model(self, image_base64: str, page_num: int, img_index: int) -> str:
        """
        Analyze an image using the vision-capable Ollama model.
        
        Args:
            image_base64 (str): Base64 encoded image
            page_num (int): Page number where image was found
            img_index (int): Index of image on the page
            
        Returns:
            str: Description of the image content
        """
        try:
            # Prepare the prompt for image analysis
            prompt = (
                "Analyze this image in detail. Focus on:\n"
                "- Any text, labels, or captions visible\n"
                "- Charts, graphs, or data visualizations\n"
                "- Diagrams, flowcharts, or technical illustrations\n"
                "- Key visual information that would be useful for document search\n"
                "- Scientific figures, mathematical equations, or formulas\n"
                "Provide a concise but comprehensive description."
            )
            
            # Make request to Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Lower temperature for more consistent descriptions
                        "top_p": 0.9
                    }
                },
                timeout=self.vision_timeout  # Configurable timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "").strip()
                
                if description:
                    logger.debug(f"✅ Analyzed image {img_index + 1} on page {page_num + 1}")
                    return description
                else:
                    logger.warning(f"⚠️ Empty response for image {img_index + 1} on page {page_num + 1}")
            else:
                logger.warning(f"⚠️ Vision API error {response.status_code} for image {img_index + 1}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️  Timeout analyzing image {img_index + 1} on page {page_num + 1}")
            self.vision_failures += 1
            self._check_disable_vision()
        except Exception as e:
            logger.warning(f"⚠️ Error analyzing image {img_index + 1} on page {page_num + 1}: {str(e)}")
            self.vision_failures += 1
            self._check_disable_vision()
        
        return ""
    
    def _check_disable_vision(self):
        """Disable vision processing if too many failures occur."""
        if self.vision_failures >= self.max_failures_before_disable and self.process_images:
            logger.warning(f"⚠️ Disabling vision processing after {self.vision_failures} failures")
            logger.info("📚 Continuing with text and table extraction only")
            self.process_images = False
    
    def _test_vision_model(self):
        """Test if the vision model is available and working."""
        try:
            logger.info(f"🧪 Testing vision model: {self.vision_model}")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": "Hello, this is a test.",
                    "stream": False
                },
                timeout=10  # Short timeout for test
            )
            
            if response.status_code == 200:
                logger.info("✅ Vision model test successful")
            else:
                logger.warning(f"⚠️ Vision model test failed with status {response.status_code}")
                if response.status_code == 404:
                    logger.warning(f"💡 Model '{self.vision_model}' not found. Install with: ollama pull {self.vision_model}")
                    logger.info("🔄 Disabling vision processing and continuing with text/tables only")
                    self.process_images = False
                    
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Vision model test timed out")
            logger.info("🔄 Disabling vision processing due to slow response")
            self.process_images = False
        except Exception as e:
            logger.warning(f"⚠️ Vision model test failed: {str(e)}")
            logger.info("🔄 Disabling vision processing and continuing with text/tables only")
            self.process_images = False
    
    def _extract_and_analyze_images(self, fitz_doc, page_num: int) -> tuple[str, int]:
        """
        Extract images from a PDF page and analyze them with vision model.
        
        Args:
            fitz_doc: PyMuPDF document object
            page_num (int): Page number (0-based)
            
        Returns:
            tuple[str, int]: (image_content_text, images_processed_count)
        """
        if not self.process_images:
            return "", 0
        
        try:
            page = fitz_doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            if not image_list:
                return "", 0
            
            logger.debug(f"🖼️ Found {len(image_list)} images on page {page_num + 1}")
            
            images_content = ""
            images_processed = 0
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    base_image = fitz_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Load image with PIL
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Skip very small images (likely decorative)
                    if image.size[0] < self.min_image_size[0] or image.size[1] < self.min_image_size[1]:
                        logger.debug(f"Skipping small image {img_index + 1} ({image.size})")
                        continue
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Resize large images to save processing time
                    max_size = (800, 800)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Convert to base64 for API
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG", optimize=True)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Analyze image with vision model
                    description = self._analyze_image_with_vision_model(
                        img_base64, page_num, img_index
                    )
                    
                    if description:
                        images_content += f"\n[IMAGE {img_index + 1} on Page {page_num + 1}]\n"
                        images_content += f"Description: {description}\n"
                        images_processed += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process image {img_index + 1} on page {page_num + 1}: {str(e)}")
                    continue
            
            if images_processed > 0:
                logger.info(f"🖼️ Processed {images_processed} images on page {page_num + 1}")
            
            return images_content, images_processed
            
        except Exception as e:
            logger.error(f"❌ Error extracting images from page {page_num + 1}: {str(e)}")
            return "", 0
    
    def load_pdfs(self) -> List[Document]:
        """
        Load all PDF files from the specified folder and extract text, tables, and images.
        
        Enhanced implementation that extracts:
        - Text content using pdfplumber (with pypdf fallback)
        - Table data using pdfplumber
        - Image analysis using PyMuPDF + vision models
        
        Returns:
            List[Document]: List of documents with text, tables, images, and metadata
        """
        logger.info(f"📚 Loading PDFs with enhanced extraction (text, tables, images) from {self.pdf_folder}")
        if self.process_images:
            logger.info(f"🖼️ Vision processing enabled with model: {self.vision_model}")
            # Test vision model availability
            self._test_vision_model()
        documents = []
        
        if not self.pdf_folder.exists():
            logger.error(f"❌ PDF folder {self.pdf_folder} does not exist!")
            return documents
        
        pdf_files = list(self.pdf_folder.glob("**/*.pdf"))
        logger.info(f"📄 Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"🔄 Processing: {pdf_file.name}")
                text_content = ""
                tables_extracted = 0
                images_extracted = 0
                total_pages = 0
                
                # Try enhanced extraction with pdfplumber + PyMuPDF first
                try:
                    # Open with both libraries for comprehensive extraction
                    with pdfplumber.open(pdf_file) as pdf_plumber:
                        fitz_doc = fitz.open(pdf_file) if self.process_images else None
                        total_pages = len(pdf_plumber.pages)
                        
                        for page_num, page in enumerate(pdf_plumber.pages):
                            page_content = f"\n--- Page {page_num + 1} ---\n"
                            
                            # Extract text
                            page_text = page.extract_text()
                            if page_text:
                                page_content += page_text
                            
                            # Extract tables
                            tables = page.extract_tables()
                            if tables:
                                logger.debug(f"📊 Found {len(tables)} tables on page {page_num + 1}")
                                for table_idx, table in enumerate(tables):
                                    if table:  # Skip empty tables
                                        table_text = self._format_table_as_text(table, table_idx + 1)
                                        if table_text.strip():
                                            page_content += table_text
                                            tables_extracted += 1
                            
                            # Extract and analyze images
                            if fitz_doc and self.process_images:
                                image_content, page_images = self._extract_and_analyze_images(fitz_doc, page_num)
                                if image_content:
                                    page_content += image_content
                                    images_extracted += page_images
                            
                            text_content += page_content
                        
                        # Clean up fitz document
                        if fitz_doc:
                            fitz_doc.close()
                            
                    logger.info(f"✅ Enhanced extraction successful: {pdf_file.name} "
                              f"({total_pages} pages, {tables_extracted} tables, {images_extracted} images)")
                
                except Exception as pdfplumber_error:
                    logger.warning(f"⚠️ pdfplumber failed for {pdf_file.name}: {pdfplumber_error}")
                    logger.info(f"🔄 Falling back to pypdf for {pdf_file.name}")
                    
                    # Fallback to pypdf for basic text extraction
                    try:
                        with open(pdf_file, 'rb') as file:
                            pdf_reader = pypdf.PdfReader(file)
                            total_pages = len(pdf_reader.pages)
                            
                            for page_num, page in enumerate(pdf_reader.pages):
                                page_text = page.extract_text()
                                if page_text:
                                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                        
                        logger.info(f"✅ Fallback extraction successful: {pdf_file.name} ({total_pages} pages)")
                    
                    except Exception as pypdf_error:
                        logger.error(f"❌ Both extraction methods failed for {pdf_file.name}: {pypdf_error}")
                        continue
                
                # Create document if we have content
                if text_content.strip():
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "source": str(pdf_file),
                            "filename": pdf_file.name,
                            "total_pages": total_pages,
                            "tables_extracted": tables_extracted,
                            "images_extracted": images_extracted,
                            "enhanced_extraction": tables_extracted > 0 or images_extracted > 0,
                            "vision_enabled": self.process_images and images_extracted > 0
                        }
                    )
                    documents.append(doc)
                    
                    # Determine extraction type based on features extracted
                    extraction_features = []
                    if tables_extracted > 0:
                        extraction_features.append(f"{tables_extracted} tables")
                    if images_extracted > 0:
                        extraction_features.append(f"{images_extracted} images")
                    
                    extraction_type = "enhanced" if extraction_features else "basic"
                    feature_text = f", {', '.join(extraction_features)}" if extraction_features else ""
                    
                    logger.info(f"📄 Loaded {pdf_file.name} ({extraction_type}, "
                              f"{total_pages} pages{feature_text})")
                else:
                    logger.warning(f"⚠️ No content extracted from {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_file.name}: {str(e)}")
        
        total_tables = sum(doc.metadata.get("tables_extracted", 0) for doc in documents)
        total_images = sum(doc.metadata.get("images_extracted", 0) for doc in documents)
        enhanced_docs = sum(1 for doc in documents if doc.metadata.get("enhanced_extraction", False))
        vision_docs = sum(1 for doc in documents if doc.metadata.get("vision_enabled", False))
        
        logger.info(f"🎉 Successfully loaded {len(documents)} documents "
                   f"({enhanced_docs} enhanced: {total_tables} tables, {total_images} images)")
        
        if self.process_images and vision_docs > 0:
            logger.info(f"🖼️ Vision analysis completed for {vision_docs} documents")
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

            logger.info(f"🔍 Result from RAG system: {result}")
            
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": []
            }
            
            # Add source information with page numbers and image references
            for doc in result.get("source_documents", []):
                # Extract page numbers from the document content
                page_numbers = self._extract_page_numbers(doc.page_content)
                page_reference = self._format_page_reference(page_numbers)
                
                # Extract image references
                image_references = self._extract_image_references(doc.page_content)
                
                # Create content preview (remove page and image markers for cleaner display)
                content_for_preview = doc.page_content
                content_for_preview = re.sub(r'\n--- Page \d+ ---\n', ' ', content_for_preview)
                content_for_preview = re.sub(r'\[IMAGE \d+ on Page \d+\]\nDescription: ', 'Image: ', content_for_preview)
                content_preview = content_for_preview[:200].strip()
                if len(content_for_preview) > 200:
                    content_preview += "..."
                
                source_info = {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page_reference": page_reference,
                    "pages": page_numbers,
                    "image_references": image_references,
                    "has_images": len(image_references) > 0,
                    "content_preview": content_preview
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
    # Parse command-line arguments
    args = parse_arguments()
    
    print("🧠 Enhanced PDF RAG System with Vision")
    print("🤝 EmbeddingGemma-300M + Ollama + Table + Image Analysis")
    print("=" * 60)
    
    try:
        # Initialize RAG system with parsed arguments
        rag_system = PDFRAG(
            pdf_folder=args.pdf_folder,
            ollama_model=args.ollama_model,
            embedding_model=args.embedding_model,
            persist_directory=args.persist_directory,
            k_similar_chunks=args.k_similar_chunks,
            enable_vision=not args.no_vision,
            vision_model=args.vision_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_image_width=args.min_image_width,
            min_image_height=args.min_image_height,
            vision_timeout=args.vision_timeout,
            max_vision_retries=args.max_vision_retries,
            max_failures_before_disable=args.max_failures_before_disable
        )
        
        print("\n⚙️ Initializing RAG system...")
        print("📥 This will download EmbeddingGemma-300M on first run...")
        print("🖼️ Vision analysis uses Gemma 3 multimodal capabilities")
        rag_system.initialize_system(force_rebuild=args.force_rebuild)
        
        print("\n🎉 Enhanced RAG system ready!")
        print("💡 Now using EmbeddingGemma for embeddings + Ollama for generation")
        print("📊 Table extraction enabled with pdfplumber")
        print("🖼️ Image analysis enabled with vision models")
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
                        # Format source with page reference if available
                        filename = source['filename']
                        page_ref = source.get('page_reference', '')
                        image_refs = source.get('image_references', [])
                        has_images = source.get('has_images', False)
                        
                        # Build reference string
                        ref_parts = []
                        if page_ref:
                            ref_parts.append(page_ref)
                        if image_refs:
                            ref_parts.append(f"{len(image_refs)} image{'s' if len(image_refs) > 1 else ''}")
                        
                        if ref_parts:
                            ref_string = f" ({', '.join(ref_parts)})"
                            print(f"  {i}. {filename}{ref_string}")
                        else:
                            print(f"  {i}. {filename}")
                        
                        # Add image indicator if present
                        if has_images:
                            print(f"     🖼️ Contains visual content: {', '.join(image_refs)}")
                        
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
        print(f"  - Ollama model is available (ollama pull {args.ollama_model})")
        print(f"  - PDF files are in the {args.pdf_folder} folder")


if __name__ == "__main__":
    main()
