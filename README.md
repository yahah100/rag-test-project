# PDF RAG System with EmbeddingGemma + Ollama

A high-performance RAG (Retrieval-Augmented Generation) system that uses Google's EmbeddingGemma-300M for embeddings and Ollama for text generation to answer questions about PDF documents.

## 🚀 System Architecture

**Hybrid Approach for Optimal Performance:**
- **EmbeddingGemma-300M**: State-of-the-art embeddings optimized for retrieval tasks
- **Ollama**: Efficient text generation with your choice of models
- **Task-Specific Prompts**: Optimized prompts for document retrieval and question answering
- **ChromaDB**: Fast vector storage and retrieval

## Features

- 📚 **Enhanced PDF Processing**: Loads and extracts text and tables from PDF files
- 📊 **Table Extraction**: Preserves table structure and data using pdfplumber
- 🧠 **Smart Chunking**: Breaks documents into optimal chunks for retrieval
- 🤖 **Ollama Integration**: Uses local Ollama models for embeddings and generation
- 💾 **Vector Storage**: Stores embeddings in ChromaDB for fast retrieval
- 🔍 **Interactive Queries**: Simple command-line interface for asking questions
- 📖 **Source Attribution**: Shows which documents were used to answer questions

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

# Install a model (recommended options)
ollama pull gemma3:4b-it-qat    # Recommended - quantized 4B model
ollama pull gemma3:1b           # Faster, smaller model
ollama pull llama3.2:1b         # Alternative lightweight option
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
2. **Enhanced Extraction**: Extracts text content and table data from each PDF page using pdfplumber
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
├── set_hf_token.py            # 🔑 HuggingFace authentication setup
├── pyproject.toml             # Dependencies and project configuration
├── chroma_db_improved/        # Vector database (created automatically)
├── .env.example              # Environment variable template
└── README.md                  # This documentation
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
  1. research_paper_1.pdf
     Preview: This paper discusses machine learning approaches...
```

### Example Questions to Try

- "What are the main research topics covered?"
- "What methodologies are used in these studies?"
- "Can you summarize the key findings?"
- "What are the limitations mentioned in the research?"
- "What future work is suggested?"

## Configuration

You can customize the system by modifying parameters in `ask_rag.py`:

```python
rag_system = ImprovedPDFRAG(
    pdf_folder="datasets",                              # Change PDF location
    ollama_model="gemma3:4b-it-qat",                   # Change Ollama model for generation
    embedding_model="google/embeddinggemma-300m",      # Change embedding model  
    persist_directory="./chroma_db_improved"           # Change database location
)
```

### Environment Variables

You can also set configuration via environment variables or `.env` file:

```bash
# .env file
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
```

## Key Classes and Functions

### `ImprovedPDFRAG` Class

Main RAG system using EmbeddingGemma for embeddings and Ollama for generation:

- `load_pdfs()` - Extracts text from PDF files using pypdf
- `chunk_documents()` - Splits text into manageable chunks with overlap
- `setup_embeddings_and_llm()` - Initializes EmbeddingGemma + Ollama components
- `create_vectorstore()` - Creates ChromaDB vector database with embeddings
- `setup_qa_chain()` - Sets up retrieval-augmented question-answering pipeline
- `query()` - Processes questions and returns answers with source attribution

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
   ollama pull gemma3:4b-it-qat
   ```

4. **"No PDFs found"**
   - Make sure PDF files are in the `datasets/` folder
   - Check file permissions and formats

5. **"Import errors"**
   ```bash
   # Install dependencies
   /home/yannik/workspace/ucla/llm-rag/.venv/bin/python -m pip install -e .
   ```

6. **"EmbeddingGemma access denied"**
   - Visit https://huggingface.co/google/embeddinggemma-300m
   - Accept Google's usage license
   - Re-run the authentication setup

### Performance Tips

- **First run setup**: Downloads EmbeddingGemma (~600MB) and builds vector database
- **Subsequent runs**: Much faster as they reuse cached models and stored embeddings
- **Model selection**: `gemma3:4b-it-qat` provides good balance of speed and quality
- **Embedding quality**: EmbeddingGemma provides superior embeddings vs general-purpose LLMs
- **Memory usage**: System uses about 1-2GB RAM for embeddings + Ollama model size
- **Storage**: Vector database grows with document collection size

## Dependencies

Key libraries used:
- `pypdf` - PDF text extraction (fallback)
- `pdfplumber` - Enhanced PDF processing with table extraction
- `langchain` - LLM framework and document processing pipeline
- `langchain-ollama` - Ollama integration for text generation
- `langchain-chroma` - ChromaDB vector store integration
- `chromadb` - Vector database for embedding storage
- `sentence-transformers` - EmbeddingGemma model integration
- `transformers` & `torch` - HuggingFace model infrastructure
- `huggingface-hub` - Authentication and model access

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

1. Add support for more file types (DOCX, TXT, HTML, etc.)
2. Implement more sophisticated chunking strategies
3. Add a web interface using Streamlit or FastAPI  
4. Implement conversation memory for multi-turn dialogues
5. Add support for multiple embedding models
6. Integrate with cloud vector databases

## License

This project is open source and available under the MIT License.
