import argparse
import logging

def parse_arguments():
    """Parse command-line arguments for the RAG system."""
    parser = argparse.ArgumentParser(
        description="Enhanced PDF RAG System with EmbeddingGemma + Ollama + Vision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--ollama-model", 
        type=str, 
        default="gemma3:4b",
        help="Ollama model for text generation (e.g., gemma3:4b, llama3.2, mistral)"
    )
    model_group.add_argument(
        "--embedding-model", 
        type=str, 
        default="google/embeddinggemma-300m",
        help="HuggingFace embedding model for document embeddings"
    )
    model_group.add_argument(
        "--vision-model", 
        type=str, 
        default="gemma3:4b",
        help="Ollama vision model for image analysis (requires multimodal model)"
    )
    
    # Processing and import arguments  
    processing_group = parser.add_argument_group("Processing Configuration")
    processing_group.add_argument(
        "--pdf-folder", 
        type=str, 
        default="datasets",
        help="Path to folder containing PDF files to process"
    )
    processing_group.add_argument(
        "--persist-directory", 
        type=str, 
        default="./chroma_db",
        help="Directory to persist ChromaDB vector store"
    )
    processing_group.add_argument(
        "--k-similar-chunks", 
        type=int, 
        default=2,
        help="Number of similar document chunks to retrieve for each query"
    )
    processing_group.add_argument(
        "--force-rebuild", 
        action="store_true",
        help="Force rebuild of vector store from PDFs (ignore existing store)"
    )
    processing_group.add_argument(
        "--no-vision", 
        action="store_true",
        help="Disable vision processing for images (text and tables only)"
    )
    
    # Text chunking arguments
    chunking_group = parser.add_argument_group("Text Chunking Configuration")
    chunking_group.add_argument(
        "--chunk-size", 
        type=int, 
        default=5_000,
        help="Maximum size of text chunks for embedding"
    )
    chunking_group.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=1000,
        help="Overlap between consecutive text chunks (typically 20%% of chunk_size)"
    )
    
    # Vision processing arguments
    vision_group = parser.add_argument_group("Vision Processing Configuration")
    vision_group.add_argument(
        "--min-image-width", 
        type=int, 
        default=50,
        help="Minimum image width in pixels (smaller images are skipped)"
    )
    vision_group.add_argument(
        "--min-image-height", 
        type=int, 
        default=50,
        help="Minimum image height in pixels (smaller images are skipped)"
    )
    vision_group.add_argument(
        "--vision-timeout", 
        type=int, 
        default=15,
        help="Timeout in seconds for vision model analysis per image"
    )
    vision_group.add_argument(
        "--max-vision-retries", 
        type=int, 
        default=1,
        help="Maximum number of retries for failed vision analysis"
    )
    vision_group.add_argument(
        "--max-failures-before-disable", 
        type=int, 
        default=5,
        help="Disable vision processing after this many failures"
    )
    
    # General arguments
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output"
    )
    general_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Enable quiet mode (less output)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be less than --chunk-size")
    
    if args.k_similar_chunks < 1:
        parser.error("--k-similar-chunks must be at least 1")
    
    if args.min_image_width < 1 or args.min_image_height < 1:
        parser.error("Minimum image dimensions must be at least 1 pixel")
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    return args