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
from input_arguments import parse_arguments

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingClass(Embeddings):
    """
    Custom embeddings class using EmbeddingGemma-300M via Ollama.
    
    This class implements task-specific prompts for optimal retrieval performance:
    - Query prompts: "task: search result | query: {text}"
    - Document prompts: "title: none | text: {text}"
    """
    
    def __init__(self, model_name: str = "embeddinggemma:300m", base_url: str = "http://localhost:11434"):
        """
        Initialize EmbeddingGemma embeddings via Ollama.
        
        Args:
            model_name (str): Ollama model name for EmbeddingGemma
            base_url (str): Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embed"
        
        logger.info(f"Initializing EmbeddingGemma via Ollama: {model_name}")
        logger.info(f"Ollama API: {self.api_url}")
        
        try:
            # Test connection to Ollama
            self._test_ollama_connection()
            logger.info("✅ EmbeddingGemma model ready via Ollama")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {str(e)}")
            self._show_ollama_help()
            raise
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available."""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": "test"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"🔐 Connected to Ollama with model: {self.model_name}")
            elif response.status_code == 404:
                raise Exception(f"Model '{self.model_name}' not found in Ollama")
            else:
                raise Exception(f"Ollama returned status code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama - is it running?")
        except requests.exceptions.Timeout:
            raise Exception("Ollama connection timeout")
            
    def _show_ollama_help(self):
        """Show Ollama setup help message."""
        logger.error("🔑 Ollama Setup Required!")
        logger.error("To use EmbeddingGemma with Ollama, you need to:")
        logger.error("1. Make sure Ollama is running: ollama serve")
        logger.error(f"2. Pull the model: ollama pull {self.model_name}")
        logger.error(f"3. Verify Ollama is accessible at: {self.base_url}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using the document-specific prompt format.
        
        According to EmbeddingGemma documentation, document embeddings should use:
        "title: {title | 'none'} | text: {content}"
        
        This enhanced version extracts title information from document content.
        
        Args:
            texts (List[str]): List of document texts to embed (may include title context)
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            # Extract title from enhanced content format: [Document: title]
            title = "none"
            content = text
            
            # Check if text starts with document title marker
            if text.startswith("[Document: ") and "]\n\n" in text:
                try:
                    # Extract title between [Document: and ]
                    title_end = text.find("]\n\n")
                    title = text[11:title_end]  # Skip "[Document: "
                    content = text[title_end + 3:]  # Skip "]\n\n"
                    
                    logger.debug(f"Extracted title: '{title}' for embedding")
                except Exception as e:
                    logger.warning(f"Failed to extract title from content: {e}")
                    # Fallback to original content
                    pass
            
            # Format with EmbeddingGemma prompt template
            formatted_text = f"title: {title} | text: {content}"
            
            # Generate embedding via Ollama API
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": formatted_text
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])
                    embeddings.append(embedding)
                else:
                    logger.error(f"Ollama API error {response.status_code} for document embedding")
                    raise Exception(f"Failed to get embedding: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error getting embedding: {str(e)}")
                raise
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Enhanced query embedding with table-awareness.
        
        According to EmbeddingGemma documentation, query embeddings should use:
        "task: search result | query: {content}"
        
        This enhanced version detects table-related queries for better retrieval.
        
        Args:
            text (str): Query text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        # Detect if query is table-related and enhance accordingly
        enhanced_query = self._enhance_query_for_tables(text)
        
        # Format text with query prompt
        formatted_text = f"task: search result | query: {enhanced_query}"
        
        # Generate embedding via Ollama API
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": formatted_text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                return embedding
            else:
                logger.error(f"Ollama API error {response.status_code} for query embedding")
                raise Exception(f"Failed to get embedding: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting query embedding: {str(e)}")
            raise
    
    def _enhance_query_for_tables(self, query: str) -> str:
        """
        Enhance query text to improve table retrieval if query appears table-related.
        
        Args:
            query (str): Original query text
            
        Returns:
            str: Enhanced query text
        """
        query_lower = query.lower()
        
        # Table-related keywords
        table_indicators = [
            'table', 'chart', 'data', 'results', 'statistics', 'numbers', 'values', 
            'comparison', 'performance', 'metrics', 'scores', 'list', 'summary',
            'figure', 'row', 'column', 'header', 'cell', 'percentage', 'rate',
            'count', 'total', 'average', 'mean', 'distribution', 'breakdown'
        ]
        
        # Question words that often precede table queries
        table_question_patterns = [
            'what are the', 'how many', 'which', 'what is the', 'compare',
            'show me', 'find', 'list', 'what were the', 'how much'
        ]
        
        is_table_query = any(indicator in query_lower for indicator in table_indicators)
        has_table_pattern = any(pattern in query_lower for pattern in table_question_patterns)
        
        if is_table_query or has_table_pattern:
            # Add table-specific context to improve retrieval
            enhanced_parts = []
            
            if is_table_query:
                enhanced_parts.append("table data")
            if 'compar' in query_lower:
                enhanced_parts.append("comparison")
            if any(word in query_lower for word in ['result', 'performance', 'score']):
                enhanced_parts.append("results")
            if any(word in query_lower for word in ['number', 'count', 'total', 'percentage']):
                enhanced_parts.append("numerical data")
            
            if enhanced_parts:
                context = " ".join(enhanced_parts)
                return f"{query} {context}"
        
        return query


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
                 embedding_model: str = "embeddinggemma:300m",
                 persist_directory: str = "./chroma_db",
                 k_similar_chunks: int = 2,
                 enable_vision: bool = True,
                 vision_model: str = "gemma3:4b",
                 chunk_size: int = 5000,
                 chunk_overlap: int = 1000,
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
        logger.info(f"🔍 Chunking: {chunk_size} chars (overlap: {chunk_overlap}) - optimized for academic papers")
        logger.info("📝 Enhanced chunking includes filename context in embeddings")
        
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
    
    def _format_table_as_text(self, table: List[List[str]], table_index: int, page_num: int = None) -> Dict[str, Any]:
        """
        Enhanced table formatting for better RAG processing with semantic understanding.
        
        Args:
            table (List[List[str]]): Table data from pdfplumber
            table_index (int): Index of the table on the page
            page_num (int): Page number for context
            
        Returns:
            Dict[str, Any]: Dictionary containing formatted text, metadata, and structure info
        """
        if not table or not any(table):
            return {"text": "", "metadata": {}, "structure": {}}
        
        try:
            # Filter out None values and convert to strings
            cleaned_table = []
            for row in table:
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cleaned_row):  # Skip completely empty rows
                    cleaned_table.append(cleaned_row)
            
            if not cleaned_table:
                return {"text": "", "metadata": {}, "structure": {}}
            
            # Extract table structure information
            num_rows = len(cleaned_table)
            num_cols = len(cleaned_table[0]) if cleaned_table else 0
            
            # Identify headers (first row) and their content
            headers = cleaned_table[0] if cleaned_table else []
            data_rows = cleaned_table[1:] if len(cleaned_table) > 1 else []
            
            # Create multiple formatted representations for better embedding
            
            # 1. Structured format (primary) - preserves full content
            structured_lines = [f"\n[TABLE {table_index}" + (f" on Page {page_num}" if page_num else "") + "]"]
            structured_lines.append(f"Table Structure: {num_rows} rows × {num_cols} columns")
            
            if headers:
                structured_lines.append("\nColumn Headers:")
                for i, header in enumerate(headers):
                    structured_lines.append(f"  Column {i+1}: {header}")
                
                structured_lines.append("\nTable Data:")
                
                # Format each row with column names for better semantic understanding
                for row_idx, row in enumerate(data_rows):
                    structured_lines.append(f"  Row {row_idx + 1}:")
                    for col_idx, (header, cell) in enumerate(zip(headers, row)):
                        if cell.strip():  # Only include non-empty cells
                            structured_lines.append(f"    {header}: {cell}")
                
                # 2. Natural language description format
                structured_lines.append("\nTable Summary:")
                if num_rows <= 2:
                    structured_lines.append(f"Small table with headers: {', '.join(headers)}")
                else:
                    structured_lines.append(f"Table containing {num_rows-1} data rows with columns: {', '.join(headers)}")
                
                # Add key-value relationships for better searchability
                structured_lines.append("\nSearchable Content:")
                for row in data_rows:
                    row_content = []
                    for header, cell in zip(headers, row):
                        if cell.strip():
                            row_content.append(f"{header} is {cell}")
                    if row_content:
                        structured_lines.append("  " + "; ".join(row_content))
            else:
                # Handle table without clear headers
                structured_lines.append("\nTable Content (no clear headers):")
                for row_idx, row in enumerate(cleaned_table):
                    row_text = " | ".join(cell for cell in row if cell.strip())
                    if row_text:
                        structured_lines.append(f"  Row {row_idx + 1}: {row_text}")
            
            structured_lines.append("")  # Empty line after table
            
            # 3. Create metadata for advanced retrieval
            table_metadata = {
                "table_index": table_index,
                "page_number": page_num,
                "dimensions": {"rows": num_rows, "columns": num_cols},
                "has_headers": bool(headers),
                "headers": headers,
                "data_types": self._analyze_table_data_types(data_rows),
                "table_type": self._classify_table_type(headers, data_rows),
                "key_terms": self._extract_table_key_terms(headers, data_rows)
            }
            
            # 4. Create structure info for chunking decisions
            table_structure = {
                "is_large": num_rows > 10 or num_cols > 5,
                "is_complex": self._is_complex_table(headers, data_rows),
                "should_keep_together": num_rows <= 20,  # Keep smaller tables together
                "estimated_tokens": self._estimate_table_tokens(cleaned_table)
            }
            
            return {
                "text": "\n".join(structured_lines),
                "metadata": table_metadata,
                "structure": table_structure,
                "raw_data": cleaned_table
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Error formatting table {table_index}: {str(e)}")
            return {
                "text": f"\n[TABLE {table_index}] - Error formatting table content\n",
                "metadata": {"error": str(e)},
                "structure": {"has_error": True}
            }
    
    def _analyze_table_data_types(self, data_rows: List[List[str]]) -> Dict[str, str]:
        """Analyze the data types in table columns for better understanding."""
        if not data_rows:
            return {}
        
        data_types = {}
        num_cols = len(data_rows[0]) if data_rows else 0
        
        for col_idx in range(num_cols):
            col_values = [row[col_idx] for row in data_rows if col_idx < len(row) and row[col_idx].strip()]
            
            if not col_values:
                data_types[f"column_{col_idx}"] = "empty"
                continue
            
            # Simple heuristic for data type detection
            numeric_count = sum(1 for val in col_values if self._is_numeric(val))
            date_count = sum(1 for val in col_values if self._looks_like_date(val))
            
            if numeric_count > len(col_values) * 0.7:
                data_types[f"column_{col_idx}"] = "numeric"
            elif date_count > len(col_values) * 0.5:
                data_types[f"column_{col_idx}"] = "date"
            else:
                data_types[f"column_{col_idx}"] = "text"
        
        return data_types
    
    def _classify_table_type(self, headers: List[str], data_rows: List[List[str]]) -> str:
        """Classify the type of table for better context."""
        if not headers:
            return "unstructured"
        
        header_text = " ".join(headers).lower()
        
        # Common table patterns
        if any(word in header_text for word in ["result", "score", "performance", "metric"]):
            return "results"
        elif any(word in header_text for word in ["name", "description", "definition"]):
            return "definitions"
        elif any(word in header_text for word in ["date", "time", "year", "month"]):
            return "temporal"
        elif any(word in header_text for word in ["price", "cost", "amount", "$", "€", "£"]):
            return "financial"
        elif any(word in header_text for word in ["count", "number", "quantity", "total"]):
            return "statistical"
        else:
            return "general"
    
    def _extract_table_key_terms(self, headers: List[str], data_rows: List[List[str]]) -> List[str]:
        """Extract key terms from table for better searchability."""
        key_terms = set()
        
        # Add headers as key terms
        for header in headers:
            key_terms.update(header.lower().split())
        
        # Add frequent terms from data (limit to avoid noise)
        term_counts = {}
        for row in data_rows:
            for cell in row:
                words = cell.lower().split()
                for word in words:
                    if len(word) > 2 and word.isalpha():  # Only meaningful words
                        term_counts[word] = term_counts.get(word, 0) + 1
        
        # Add top frequent terms
        frequent_terms = [term for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
        key_terms.update(frequent_terms)
        
        return list(key_terms)
    
    def _is_complex_table(self, headers: List[str], data_rows: List[List[str]]) -> bool:
        """Determine if table has complex structure requiring special handling."""
        if not headers or not data_rows:
            return False
        
        # Complex if many columns or irregular structure
        num_cols = len(headers)
        if num_cols > 6:
            return True
        
        # Check for irregular row lengths
        irregular_rows = sum(1 for row in data_rows if len(row) != num_cols)
        if irregular_rows > len(data_rows) * 0.3:
            return True
        
        # Check for nested headers or multi-line cells
        for header in headers:
            if '\n' in header or len(header) > 50:
                return True
        
        return False
    
    def _estimate_table_tokens(self, table: List[List[str]]) -> int:
        """Estimate token count for table content."""
        total_chars = sum(len(cell) for row in table for cell in row)
        return total_chars // 4  # Rough estimate of tokens
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a string represents a numeric value."""
        try:
            float(value.replace(',', '').replace('$', '').replace('%', ''))
            return True
        except ValueError:
            return False
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if a string looks like a date."""
        import re
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # MM/DD/YYYY, MM-DD-YY, etc.
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',    # YYYY-MM-DD
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # Month names
            r'\d{4}',  # Just year
        ]
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in date_patterns)
    
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
                table_metadata_list = []  # Store table metadata for enhanced retrieval
                
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
                            
                            # Extract tables with enhanced processing
                            tables = page.extract_tables()
                            if tables:
                                logger.debug(f"📊 Found {len(tables)} tables on page {page_num + 1}")
                                for table_idx, table in enumerate(tables):
                                    if table:  # Skip empty tables
                                        table_result = self._format_table_as_text(table, table_idx + 1, page_num + 1)
                                        if table_result["text"].strip():
                                            page_content += table_result["text"]
                                            tables_extracted += 1
                                            
                                            # Store table metadata for enhanced retrieval
                                            table_metadata_list.append(table_result["metadata"])
                            
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
                            "vision_enabled": self.process_images and images_extracted > 0,
                            "table_metadata": table_metadata_list,  # Enhanced table information
                            "has_structured_data": len(table_metadata_list) > 0
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
        Enhanced chunking with table-aware splitting to preserve table structure.
        
        This method:
        - Identifies tables in document content
        - Keeps small tables together in single chunks
        - Handles large tables by creating dedicated chunks
        - Preserves table context and metadata
        
        Args:
            documents (List[Document]): List of documents to chunk
            
        Returns:
            List[Document]: List of chunked documents with enhanced metadata
        """
        logger.info("✂️ Enhanced chunking with table-aware processing")
        
        chunked_docs = []
        
        for document in documents:
            # Extract filename for title (remove extension and path)
            filename = document.metadata.get("filename", "Unknown")
            title = Path(filename).stem if filename != "Unknown" else "Unknown Document"
            table_metadata_list = document.metadata.get("table_metadata", [])
            
            # Use table-aware chunking strategy
            doc_chunks = self._chunk_document_with_table_awareness(document, table_metadata_list)
            
            # Enhance each chunk with filename context for embeddings
            for chunk_idx, chunk in enumerate(doc_chunks):
                # Store the title information that will be used in embeddings
                chunk.metadata["title"] = title
                chunk.metadata["filename_for_embedding"] = title
                chunk.metadata["chunk_index"] = chunk_idx
                
                # Add table-specific metadata if this chunk contains tables
                chunk_tables = self._identify_tables_in_chunk(chunk.page_content, table_metadata_list)
                if chunk_tables:
                    chunk.metadata["contains_tables"] = True
                    chunk.metadata["table_info"] = chunk_tables
                    chunk.metadata["chunk_type"] = "table_rich"
                else:
                    chunk.metadata["contains_tables"] = False
                    chunk.metadata["chunk_type"] = "text"
                
                # Optionally prepend filename context to content for better embeddings
                # This helps the embedding model understand document context
                enhanced_content = f"[Document: {title}]\n\n{chunk.page_content}"
                chunk.page_content = enhanced_content
                
                chunked_docs.append(chunk)
        
        table_chunks = sum(1 for doc in chunked_docs if doc.metadata.get("contains_tables", False))
        logger.info(f"📦 Created {len(chunked_docs)} chunks from {len(documents)} documents")
        logger.info(f"📊 {table_chunks} chunks contain table data with preserved structure")
        logger.info("📝 Enhanced chunks with filename context for better embeddings")
        return chunked_docs
    
    def _chunk_document_with_table_awareness(self, document: Document, table_metadata_list: List[Dict]) -> List[Document]:
        """
        Chunk a document while preserving table structure integrity.
        
        Args:
            document (Document): Document to chunk
            table_metadata_list (List[Dict]): List of table metadata
            
        Returns:
            List[Document]: List of table-aware chunks
        """
        content = document.page_content
        
        # If no tables, use standard chunking
        if not table_metadata_list:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            return text_splitter.split_documents([document])
        
        # Table-aware chunking
        chunks = []
        current_pos = 0
        
        # Find all table positions in the document
        table_positions = []
        for table_meta in table_metadata_list:
            table_marker = f"[TABLE {table_meta.get('table_index', 0)}"
            if table_meta.get('page_number'):
                table_marker += f" on Page {table_meta['page_number']}"
            table_marker += "]"
            
            start_pos = content.find(table_marker, current_pos)
            if start_pos != -1:
                # Find end of table
                next_table_start = len(content)
                for other_meta in table_metadata_list:
                    other_marker = f"[TABLE {other_meta.get('table_index', 0)}"
                    if other_meta.get('page_number'):
                        other_marker += f" on Page {other_meta['page_number']}"
                    other_marker += "]"
                    
                    other_pos = content.find(other_marker, start_pos + 1)
                    if other_pos != -1 and other_pos < next_table_start:
                        next_table_start = other_pos
                
                # Look for next page marker or image marker as potential end
                next_page_pos = content.find("\n--- Page", start_pos + 1)
                next_image_pos = content.find("\n[IMAGE", start_pos + 1)
                
                potential_ends = [pos for pos in [next_table_start, next_page_pos, next_image_pos] if pos > start_pos]
                end_pos = min(potential_ends) if potential_ends else len(content)
                
                table_positions.append({
                    'start': start_pos,
                    'end': end_pos,
                    'metadata': table_meta,
                    'size': end_pos - start_pos
                })
        
        # Sort table positions by start position
        table_positions.sort(key=lambda x: x['start'])
        
        # Create chunks with table awareness
        current_pos = 0
        
        for table_pos in table_positions:
            # Add pre-table content if any
            if current_pos < table_pos['start']:
                pre_table_content = content[current_pos:table_pos['start']].strip()
                if pre_table_content:
                    # Use standard chunking for pre-table content
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    temp_doc = Document(page_content=pre_table_content, metadata=document.metadata.copy())
                    pre_chunks = text_splitter.split_documents([temp_doc])
                    chunks.extend(pre_chunks)
            
            # Handle table content
            table_content = content[table_pos['start']:table_pos['end']].strip()
            if table_content:
                # Decide whether to keep table in one chunk or split
                table_size = table_pos['size']
                
                if table_size <= self.chunk_size * 1.5:  # Keep smaller tables together
                    table_chunk = Document(
                        page_content=table_content,
                        metadata=document.metadata.copy()
                    )
                    chunks.append(table_chunk)
                else:
                    # Large table - split but try to keep logical sections together
                    logger.info(f"📊 Large table detected ({table_size} chars), using careful splitting")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    temp_doc = Document(page_content=table_content, metadata=document.metadata.copy())
                    table_chunks = text_splitter.split_documents([temp_doc])
                    chunks.extend(table_chunks)
            
            current_pos = table_pos['end']
        
        # Add remaining content after last table
        if current_pos < len(content):
            remaining_content = content[current_pos:].strip()
            if remaining_content:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                temp_doc = Document(page_content=remaining_content, metadata=document.metadata.copy())
                final_chunks = text_splitter.split_documents([temp_doc])
                chunks.extend(final_chunks)
        
        return chunks if chunks else [document]  # Fallback to original if something went wrong
    
    def _identify_tables_in_chunk(self, chunk_content: str, table_metadata_list: List[Dict]) -> List[Dict]:
        """
        Identify which tables are present in a given chunk.
        
        Args:
            chunk_content (str): Content of the chunk
            table_metadata_list (List[Dict]): List of all table metadata
            
        Returns:
            List[Dict]: List of table metadata for tables in this chunk
        """
        chunk_tables = []
        
        for table_meta in table_metadata_list:
            table_marker = f"[TABLE {table_meta.get('table_index', 0)}"
            if table_meta.get('page_number'):
                table_marker += f" on Page {table_meta['page_number']}"
            table_marker += "]"
            
            if table_marker in chunk_content:
                # Create a summary of table info for chunk metadata
                table_summary = {
                    "table_index": table_meta.get('table_index'),
                    "page_number": table_meta.get('page_number'),
                    "table_type": table_meta.get('table_type', 'general'),
                    "headers": table_meta.get('headers', []),
                    "dimensions": table_meta.get('dimensions', {}),
                    "key_terms": table_meta.get('key_terms', [])[:5]  # Limit for metadata
                }
                chunk_tables.append(table_summary)
        
        return chunk_tables
    
    def _extract_table_references(self, content: str) -> List[str]:
        """Extract table references from chunk content."""
        import re
        
        # Find table markers like "[TABLE 1 on Page 3]"
        table_pattern = r'\[TABLE (\d+)(?: on Page (\d+))?\]'
        matches = re.findall(table_pattern, content)
        
        # Format as readable references
        table_refs = []
        for table_num, page_num in matches:
            if page_num:
                table_refs.append(f"Table {table_num} (p. {page_num})")
            else:
                table_refs.append(f"Table {table_num}")
        
        return table_refs
    
    def _create_table_aware_preview(self, content: str, table_info: List[Dict]) -> str:
        """Create a preview that highlights table content and structure."""
        if not table_info:
            preview = content[:200].strip()
            return preview + "..." if len(content) > 200 else preview
        
        # Create a structured preview for table-rich content
        preview_parts = []
        
        for table_meta in table_info:
            table_type = table_meta.get('table_type', 'general')
            headers = table_meta.get('headers', [])
            dimensions = table_meta.get('dimensions', {})
            
            table_desc = f"Table {table_meta.get('table_index', '')}" 
            if table_meta.get('page_number'):
                table_desc += f" (p. {table_meta['page_number']})"
            
            table_desc += f": {table_type} table"
            
            if dimensions:
                rows = dimensions.get('rows', 0)
                cols = dimensions.get('columns', 0)
                table_desc += f" ({rows}×{cols})"
            
            if headers:
                table_desc += f" - Headers: {', '.join(headers[:3])}"
                if len(headers) > 3:
                    table_desc += "..."
            
            preview_parts.append(table_desc)
        
        # Add some actual content
        clean_content = re.sub(r'\[TABLE \d+[^\]]*\]', '', content)
        clean_content = re.sub(r'Table Structure:[^\n]*\n', '', clean_content)
        clean_content = re.sub(r'Column Headers:[^\n]*\n', '', clean_content)
        content_sample = clean_content[:100].strip()
        
        if content_sample:
            preview_parts.append(f"Content: {content_sample}...")
        
        return " | ".join(preview_parts)
    
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
            self.embeddings = EmbeddingClass(model_name=self.embedding_model)
            
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
        
        # Enhanced prompt template with table awareness
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
- When referencing table data, preserve the structure and relationships between data points
- For numerical data from tables, include specific values and their context (headers, units, etc.)
- If comparing data from tables, clearly indicate which values are being compared

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
            
            # Add enhanced source information with table, page, and image references
            for doc in result.get("source_documents", []):
                # Extract page numbers from the document content
                page_numbers = self._extract_page_numbers(doc.page_content)
                page_reference = self._format_page_reference(page_numbers)
                
                # Extract image references
                image_references = self._extract_image_references(doc.page_content)
                
                # Extract table references and metadata
                table_references = self._extract_table_references(doc.page_content)
                table_info = doc.metadata.get("table_info", [])
                
                # Create content preview (remove markers for cleaner display)
                content_for_preview = doc.page_content
                content_for_preview = re.sub(r'\[Document: [^\]]+\]\n\n', '', content_for_preview)  # Remove document title
                content_for_preview = re.sub(r'\n--- Page \d+ ---\n', ' ', content_for_preview)
                content_for_preview = re.sub(r'\[IMAGE \d+ on Page \d+\]\nDescription: ', 'Image: ', content_for_preview)
                
                # For table content, provide more structured preview
                if table_info:
                    content_preview = self._create_table_aware_preview(content_for_preview, table_info)
                else:
                    content_preview = content_for_preview[:200].strip()
                    if len(content_for_preview) > 200:
                        content_preview += "..."
                
                source_info = {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page_reference": page_reference,
                    "pages": page_numbers,
                    "image_references": image_references,
                    "has_images": len(image_references) > 0,
                    "table_references": table_references,
                    "table_info": table_info,
                    "has_tables": len(table_references) > 0 or len(table_info) > 0,
                    "chunk_type": doc.metadata.get("chunk_type", "text"),
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
    print("🤝 EmbeddingGemma-300M via Ollama + Table + Image Analysis")
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
        print("📥 Using Ollama for EmbeddingGemma-300M embeddings...")
        print("🖼️ Vision analysis uses Gemma 3 multimodal capabilities")
        rag_system.initialize_system(force_rebuild=args.force_rebuild)
        
        print("\n🎉 Enhanced RAG system ready!")
        print("💡 Now using EmbeddingGemma via Ollama for embeddings + Ollama for generation")
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
                        table_refs = source.get('table_references', [])
                        has_tables = source.get('has_tables', False)
                        chunk_type = source.get('chunk_type', 'text')
                        
                        # Build reference string
                        ref_parts = []
                        if page_ref:
                            ref_parts.append(page_ref)
                        if image_refs:
                            ref_parts.append(f"{len(image_refs)} image{'s' if len(image_refs) > 1 else ''}")
                        if table_refs:
                            ref_parts.append(f"{len(table_refs)} table{'s' if len(table_refs) > 1 else ''}")
                        
                        if ref_parts:
                            ref_string = f" ({', '.join(ref_parts)})"
                            print(f"  {i}. {filename}{ref_string}")
                        else:
                            print(f"  {i}. {filename}")
                        
                        # Add table indicator if present
                        if has_tables:
                            print(f"     📊 Contains table data: {', '.join(table_refs)}")
                        
                        # Add image indicator if present
                        if has_images:
                            print(f"     🖼️ Contains visual content: {', '.join(image_refs)}")
                        
                        # Add chunk type indicator for table-rich content
                        if chunk_type == "table_rich":
                            print(f"     📈 Table-rich content with structured data")
                        
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
        print("  - Ollama is running (ollama serve)")
        print(f"  - Embedding model is available (ollama pull {args.embedding_model})")
        print(f"  - Ollama LLM model is available (ollama pull {args.ollama_model})")
        print(f"  - PDF files are in the {args.pdf_folder} folder")


if __name__ == "__main__":
    main()
