# Enhanced PDF RAG System with Vision

A high-performance RAG (Retrieval-Augmented Generation) system that uses Google's EmbeddingGemma-300M for embeddings, Ollama for text generation, and vision models for comprehensive PDF analysis including text, tables, and images.

## 🚀 System Architecture

**Hybrid Approach for Optimal Performance:**
- **EmbeddingGemma-300M**: State-of-the-art embeddings optimized for retrieval tasks
- **Ollama**: Efficient text generation with your choice of models
- **Task-Specific Prompts**: Optimized prompts for document retrieval and question answering
- **ChromaDB**: Fast vector storage and retrieval

## Features

- 📚 **Enhanced PDF Processing**: Loads and extracts text, tables, and images from PDF files
- 📊 **Table Extraction**: Preserves table structure and data using pdfplumber
- 🖼️ **Vision Analysis**: Analyzes images, charts, graphs, and diagrams using Gemma 3 multimodal models
- 📄 **Page Reference Tracking**: Automatically tracks and displays page numbers for all sources
- 🎯 **Multi-Content Search**: Search across text, tables, and image descriptions simultaneously
- 🧠 **Smart Chunking**: Breaks documents into optimal chunks for retrieval
- 🤖 **Ollama Integration**: Uses local Ollama models for embeddings and generation
- 💾 **Vector Storage**: Stores embeddings in ChromaDB for fast retrieval
- 🔍 **Interactive Queries**: Simple command-line interface for asking questions
- 📖 **Comprehensive Source Attribution**: Shows documents, pages, and visual content references

## Quick Start

### 1. Prerequisites

Make sure you have:
- Python 3.13+ (already configured with virtual environment)
- [Ollama](https://ollama.ai/) installed and running
- PDF files in the `datasets/` folder (already present)
- HuggingFace account with access to EmbeddingGemma

### 2. Install Ollama Model

First, start Ollama and install a suitable model:

```bash
# Start Ollama (in a separate terminal)
ollama serve

# Install models (recommended options)
ollama pull gemma3:4b-it-qat    # Recommended - quantized 4B model with vision
ollama pull gemma3:12b          # Better quality vision analysis (requires more memory)
ollama pull gemma3:1b           # Faster, smaller model (limited vision capabilities)

# Alternative vision models (optional)
ollama pull llava:7b            # Dedicated vision model
ollama pull llava:13b           # Higher quality vision model
```

### 3. Set Up HuggingFace Authentication

**Required for EmbeddingGemma access:**

1. **Accept the License**:
   - Visit: https://huggingface.co/google/embeddinggemma-300m
   - Log in and accept Google's usage license

2. **Get Your Token**:
   - Go to: https://huggingface.co/settings/tokens
   - Create or copy a **Read** token

3. **Authenticate Locally**:
   ```bash
   # Run the authentication setup
   /home/yannik/workspace/ucla/llm-rag/.venv/bin/python set_hf_token.py
   ```
   - Paste your token when prompted
   - Token will be saved for future use

### 4. Run the RAG System

```bash
# Run the interactive RAG system
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py
```

## How It Works

The RAG system uses a hybrid approach for optimal performance:

### System Architecture (EmbeddingGemma + Ollama)

1. **Document Loading**: Reads PDF files from the `datasets/` folder
2. **Multi-Modal Extraction**: 
   - Text content using pdfplumber (with pypdf fallback)
   - Table data preservation using pdfplumber  
   - Image extraction and analysis using PyMuPDF + Gemma 3 vision
3. **Smart Chunking**: Splits documents into 1000-character chunks with 200-character overlap
4. **Embedding Generation**: Uses **EmbeddingGemma-300M** with task-specific prompts:
   - Documents: `"title: none | text: {content}"`
   - Queries: `"task: search result | query: {question}"`
5. **Vector Storage**: Saves embeddings in ChromaDB for fast retrieval
6. **Query Processing**:
   - User question → EmbeddingGemma creates query embedding
   - Similarity search finds relevant document chunks
   - **Ollama** generates answer using retrieved context  
   - Returns answer with source attribution

## File Structure

```
llm-rag/
├── datasets/                    # PDF files (your research papers)
├── ask_rag.py                  # 🚀 Main RAG system (EmbeddingGemma + Ollama)
├── input_arguments.py          # ⚙️ Command-line argument parsing and configuration
├── set_hf_token.py            # 🔑 HuggingFace authentication setup
├── pyproject.toml             # Dependencies and project configuration
├── chroma_db/                 # Vector database (created automatically)
├── .env.example              # Environment variable template
└── README.md                  # This documentation
```

## Command-Line Interface

The RAG system now features a comprehensive command-line interface for easy configuration. All parameters can be customized without modifying code.

### Quick Reference

```bash
# View all available options
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py --help
```

### Command-Line Arguments

**Model Configuration:**
- `--ollama-model` - Ollama model for text generation (default: `gemma3:4b`)
- `--embedding-model` - HuggingFace embedding model (default: `google/embeddinggemma-300m`)
- `--vision-model` - Ollama vision model for image analysis (default: `gemma3:4b`)

**Processing Configuration:**
- `--pdf-folder` - Path to PDF files directory (default: `datasets`)
- `--persist-directory` - ChromaDB storage location (default: `./chroma_db`)
- `--k-similar-chunks` - Number of document chunks to retrieve (default: `2`)
- `--force-rebuild` - Force rebuild of vector store from PDFs
- `--no-vision` - Disable vision processing for faster operation

**Text Chunking:**
- `--chunk-size` - Maximum chunk size in characters (default: `1000`)
- `--chunk-overlap` - Overlap between chunks in characters (default: `200`)

**Vision Processing:**
- `--min-image-width/height` - Minimum image dimensions in pixels (default: `50x50`)
- `--vision-timeout` - Timeout per image analysis in seconds (default: `15`)
- `--max-vision-retries` - Retries for failed vision analysis (default: `1`)
- `--max-failures-before-disable` - Disable vision after failures (default: `5`)

**General Options:**
- `--verbose/-v` - Enable verbose logging
- `--quiet/-q` - Enable quiet mode
- `--help/-h` - Show help message

## CLI Usage Examples

### Basic Usage Scenarios

```bash
# Default configuration - recommended for most users
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py

# Fast mode - disable vision processing for speed
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py --no-vision

# High-quality mode - use larger models
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --ollama-model gemma3:7b \
    --vision-model gemma3:7b
```

### Advanced Configuration Examples

```bash
# Custom PDF folder and database location
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --pdf-folder ./my-papers \
    --persist-directory ./my-vector-db

# Optimize for large documents
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --chunk-size 1500 \
    --chunk-overlap 300 \
    --k-similar-chunks 4

# Force rebuild with different models
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --force-rebuild \
    --ollama-model llama3.2 \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Vision-optimized settings
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --vision-model gemma3:7b \
    --vision-timeout 30 \
    --min-image-width 100 \
    --min-image-height 100

# Debugging and development
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --verbose \
    --force-rebuild
```

### Production Deployment Examples

```bash
# Memory-optimized for server deployment
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --ollama-model gemma3:1b \
    --no-vision \
    --chunk-size 800 \
    --quiet

# High-throughput configuration
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --k-similar-chunks 1 \
    --vision-timeout 10 \
    --max-failures-before-disable 2
```

## Usage Examples

### Interactive Mode
```bash
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py

🧠 RAG System (EmbeddingGemma + Ollama)
🔍 Ask a question: What are the main topics in these research papers?

🤔 Thinking...
📝 Answer: Based on the documents, the main topics include...

📚 Sources:
  1. research_paper_1.pdf (pp. 3-5, 2 images)
     🖼️ Contains visual content: Image 1 (p. 4), Image 2 (p. 5)
     Preview: This paper discusses machine learning approaches...
  
  2. methodology_paper.pdf (p. 12)
     Preview: Image: Bar chart showing accuracy scores across different models...
```

### Page Reference System

The enhanced RAG system automatically tracks and displays page numbers for all source citations:

- **Single page**: `paper.pdf (p. 5)`
- **Page range**: `paper.pdf (pp. 3-5)`
- **Multiple pages**: `paper.pdf (pp. 2, 4-6)`
- **Multiple sources**: Each source shows its own page references

This makes it easy to find and verify information in the original documents.

### Example Questions to Try

- "What are the main research topics covered?"
- "What methodologies are used in these studies?" 
- "Can you summarize the key findings?"
- "What are the limitations mentioned in the research?"
- "What future work is suggested?"
- "Show me the results from the performance comparison table"
- "Describe the charts and graphs in the results section"
- "What does Figure 3 show about the system architecture?"
- "Explain the data visualization on page 5"
- "What trends are visible in the performance graphs?"

## Configuration

You can customize the system by modifying parameters in `ask_rag.py`:

### Command-Line Arguments (Recommended)

The system now supports comprehensive command-line configuration:

```bash
# Basic usage with default settings
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py

# Custom model configuration
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --ollama-model llama3.2 \
    --embedding-model google/embeddinggemma-300m \
    --vision-model gemma3:4b

# Performance and processing options
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --chunk-size 1500 \
    --chunk-overlap 300 \
    --k-similar-chunks 3 \
    --no-vision

# Force rebuild database with verbose output
/home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
    --force-rebuild \
    --verbose
```

### Programmatic Configuration (Advanced)

You can also customize the system by modifying parameters in code:

```python
rag_system = PDFRAG(
    pdf_folder="datasets",                              # Change PDF location
    ollama_model="gemma3:4b",                          # Change Ollama model for generation
    embedding_model="google/embeddinggemma-300m",      # Change embedding model  
    persist_directory="./chroma_db"                    # Change database location
)
```

### Environment Variables

You can also set configuration via environment variables or `.env` file:

```bash
# .env file
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
```

## Key Classes and Functions

### Main Components

**`input_arguments.py`** - Command-line interface module:
- `parse_arguments()` - Parses and validates all command-line arguments
- Organized argument groups: Model Configuration, Processing Configuration, Text Chunking, Vision Processing, General Options
- Built-in validation for parameter combinations and value ranges
- Automatic logging level configuration based on verbosity flags

**`ask_rag.py`** - Main application module:

### `PDFRAG` Class

Main RAG system using EmbeddingGemma for embeddings and Ollama for generation:

- `__init__()` - Initializes system with configurable parameters for models, chunking, vision processing
- `load_pdfs()` - Enhanced PDF extraction with text, tables, and image analysis
- `chunk_documents()` - Splits text into configurable chunks with overlap
- `setup_embeddings_and_llm()` - Initializes EmbeddingGemma + Ollama components
- `create_vectorstore()` - Creates ChromaDB vector database with embeddings
- `setup_qa_chain()` - Sets up retrieval-augmented question-answering pipeline  
- `query()` - Processes questions and returns answers with source attribution
- `_extract_and_analyze_images()` - Vision processing for PDF images
- `_format_table_as_text()` - Table extraction and formatting

### `EmbeddingGemmaEmbeddings` Class

Custom embeddings implementation with task-specific prompts:
- `embed_documents()` - Creates embeddings for documents using document prompts
- `embed_query()` - Creates embeddings for queries using query prompts
- `_check_hf_auth()` - Verifies HuggingFace authentication
- `_show_auth_help()` - Displays authentication setup instructions

## Troubleshooting

### Common Issues

1. **"HuggingFace Authentication Required"**
   ```bash
   # Run authentication setup
   /home/yannik/workspace/ucla/llm-rag/.venv/bin/python set_hf_token.py
   ```
   - Make sure you've accepted the EmbeddingGemma license
   - Verify your token has 'Read' permissions

2. **"Ollama not running"**
   ```bash
   # Start Ollama
   ollama serve
   ```

3. **"Model not found"**
   ```bash
   # Install recommended model
   ollama pull gemma3:4b
   
   # Or specify a different model via command line
   /home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py --ollama-model llama3.2
   ```

4. **"No PDFs found"**
   ```bash
   # Default location
   - Make sure PDF files are in the `datasets/` folder
   
   # Custom location via command line
   /home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py --pdf-folder ./my-papers
   ```

5. **"Import errors"**
   ```bash
   # Install dependencies
   /home/yannik/workspace/ucla/llm-rag/.venv/bin/python -m pip install -e .
   ```

6. **"EmbeddingGemma access denied"**
   - Visit https://huggingface.co/google/embeddinggemma-300m
   - Accept Google's usage license
   - Re-run the authentication setup

7. **"Vision processing failures"**
   ```bash
   # Disable vision for faster processing
   /home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py --no-vision
   
   # Adjust vision timeout settings
   /home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py --vision-timeout 30
   ```

8. **"Memory issues"**
   ```bash
   # Use smaller models and chunks
   /home/yannik/workspace/ucla/llm-rag/.venv/bin/python ask_rag.py \
       --ollama-model gemma3:1b \
       --chunk-size 800 \
       --no-vision
   ```

### Performance Tips

- **First run setup**: Downloads EmbeddingGemma (~600MB) and builds vector database
- **Subsequent runs**: Much faster as they reuse cached models and stored embeddings
- **Model selection**: Configure via `--ollama-model` - `gemma3:4b` provides good balance of speed and quality
- **Speed optimization**: Use `--no-vision` to disable image processing for faster operation
- **Memory optimization**: Use `--ollama-model gemma3:1b` and `--chunk-size 800` for lower memory usage
- **Embedding quality**: EmbeddingGemma provides superior embeddings vs general-purpose LLMs
- **Memory usage**: System uses about 1-2GB RAM for embeddings + Ollama model size
- **Storage**: Vector database grows with document collection size
- **Force rebuild**: Use `--force-rebuild` when changing embedding models or chunk parameters
- **Chunking optimization**: Larger `--chunk-size` for longer documents, adjust `--chunk-overlap` accordingly

## Dependencies

Key libraries used:
- `argparse` - Command-line argument parsing (built-in Python module)
- `pypdf` - PDF text extraction (fallback)
- `pdfplumber` - Enhanced PDF processing with table extraction
- `pymupdf` (fitz) - Image extraction from PDFs
- `pillow` - Image processing and manipulation
- `requests` - API communication with Ollama vision models
- `langchain` - LLM framework and document processing pipeline
- `langchain-ollama` - Ollama integration for text generation
- `langchain-chroma` - ChromaDB vector store integration
- `chromadb` - Vector database for embedding storage
- `sentence-transformers` - EmbeddingGemma model integration
- `transformers` & `torch` - HuggingFace model infrastructure
- `huggingface-hub` - Authentication and model access

### Modular Architecture
- `input_arguments.py` - Centralized command-line interface with comprehensive argument validation
- `ask_rag.py` - Main application logic with configurable components

## Authentication Details

### HuggingFace Setup

1. **Create Account**: Sign up at https://huggingface.co if you don't have an account
2. **Accept License**: Visit the EmbeddingGemma model page and accept terms
3. **Generate Token**: Create a Read token in your HuggingFace settings
4. **Local Setup**: Use `set_hf_token.py` to configure authentication

### Token Management

```bash
# Set token manually
python set_hf_token.py

# Or set environment variable
export HUGGINGFACE_HUB_TOKEN=your_token_here

# Or use .env file
echo "HUGGINGFACE_HUB_TOKEN=your_token_here" > .env
```

## Next Steps

To extend this system, you could:

1. **File format support**: Add support for more file types (DOCX, TXT, HTML, etc.)
2. **Advanced chunking**: Implement semantic chunking or document-structure-aware splitting  
3. **Web interface**: Add a web UI using Streamlit or FastAPI
4. **Conversation memory**: Implement multi-turn dialogue capabilities
5. **Model flexibility**: Add support for multiple embedding models simultaneously
6. **Cloud integration**: Integrate with cloud vector databases (Pinecone, Weaviate, etc.)
7. **Configuration management**: Extend `input_arguments.py` with config files (YAML/JSON)
8. **API mode**: Add REST API endpoints for programmatic access
9. **Batch processing**: Add batch document processing capabilities
10. **Custom prompts**: Make prompt templates configurable via command line

## License

This project is open source and available under the MIT License.
