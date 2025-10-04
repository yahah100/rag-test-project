import asyncio
import aiohttp
import os
import sys
import traceback
from pathlib import Path
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "gemma3:4b"  # Change this to your preferred model
OLLAMA_VISION_MODEL = "llava:7b"  # Change this to your vision model
OLLAMA_EMBED_MODEL = "embeddinggemma:300m"  # Change this to your embedding model

# Paths
DATASETS_DIR = "./datasets"
OUTPUT_DIR = "./output"
RAG_STORAGE_DIR = "./rag_storage"


async def ollama_complete(model, prompt, system_prompt=None, history_messages=None, 
                          image_data=None, messages=None, **kwargs):
    """
    Call Ollama API for text completion (async version)
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    
    # Build messages
    msg_list = []
    
    # Handle messages format (for multimodal VLM enhanced query)
    if messages:
        msg_list = messages
    else:
        # Add system prompt if provided
        if system_prompt:
            msg_list.append({"role": "system", "content": system_prompt})
        
        # Add history messages if provided
        if history_messages:
            msg_list.extend(history_messages)
        
        # Add current prompt with optional image
        if image_data:
            msg_list.append({
                "role": "user",
                "content": prompt,
                "images": [image_data]  # Ollama expects base64 image data
            })
        else:
            msg_list.append({
                "role": "user",
                "content": prompt
            })
    
    payload = {
        "model": model,
        "messages": msg_list,
        "stream": False,
        "options": kwargs.get("options", {})
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                response.raise_for_status()
                result = await response.json()
                return result["message"]["content"]
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        raise


async def ollama_embed(texts, model):
    """
    Call Ollama API for embeddings (async version)
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    
    embeddings = []
    
    async with aiohttp.ClientSession() as session:
        for text in texts:
            payload = {
                "model": model,
                "input": text
            }
            
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    response.raise_for_status()
                    result = await response.json()
                    # Ollama returns {"embeddings": [[...]]} - take the first embedding
                    embeddings.append(result["embeddings"][0])
            except Exception as e:
                print(f"Error getting embeddings from Ollama: {e}")
                raise
    
    return embeddings


async def main():
    """
    Main function to process all PDFs in the datasets directory using RAG-Anything
    """
    
    # Add venv bin to PATH so mineru command can be found
    venv_bin = os.path.join(os.path.dirname(sys.executable), '')
    os.environ['PATH'] = venv_bin + os.pathsep + os.environ.get('PATH', '')
    
    print("=" * 80)
    print("RAG-Anything with Ollama - PDF Processing")
    print("=" * 80)
    print(f"Ollama Base URL: {OLLAMA_BASE_URL}")
    print(f"LLM Model: {OLLAMA_LLM_MODEL}")
    print(f"Vision Model: {OLLAMA_VISION_MODEL}")
    print(f"Embedding Model: {OLLAMA_EMBED_MODEL}")
    print(f"Datasets Directory: {DATASETS_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"RAG Storage Directory: {RAG_STORAGE_DIR}")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAG_STORAGE_DIR, exist_ok=True)
    
    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir=RAG_STORAGE_DIR,
        parser="mineru",  # Use MinerU parser
        parse_method="auto",  # Auto-detect parsing method
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # Define LLM model function for Ollama
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await ollama_complete(
            OLLAMA_LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    # Define vision model function for Ollama
    async def vision_model_func(prompt, system_prompt=None, history_messages=[], 
                                image_data=None, messages=None, **kwargs):
        # Use vision model if image data is present
        if image_data or messages:
            return await ollama_complete(
                OLLAMA_VISION_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                image_data=image_data,
                messages=messages,
                **kwargs
            )
        else:
            # Fall back to regular LLM for text-only
            return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)
    
    # Define embedding function for Ollama
    # Get embedding dimension from the first embedding
    test_embed = await ollama_embed(["test"], OLLAMA_EMBED_MODEL)
    embedding_dim = len(test_embed[0])
    print(f"Embedding dimension: {embedding_dim}")
    
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=lambda texts: ollama_embed(texts, OLLAMA_EMBED_MODEL)
    )
    
    # Initialize RAGAnything
    print("\nInitializing RAGAnything...")
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    
    print("RAGAnything initialized successfully!")
    
    # Find all PDF files in datasets directory
    print(f"\nSearching for PDF files in {DATASETS_DIR}...")
    datasets_path = Path(DATASETS_DIR)
    pdf_files = list(datasets_path.rglob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files")
    
    if len(pdf_files) == 0:
        print("No PDF files found. Exiting.")
        return
    
    # Ask user if they want to process all or select specific files
    print("\nOptions:")
    print("1. Process all PDFs")
    print("2. Process only the first 5 PDFs (for testing)")
    print("3. Process a specific number of PDFs")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "2":
        pdf_files = pdf_files[:5]
        print(f"Processing first 5 PDFs")
    elif choice == "3":
        num = int(input("How many PDFs to process? "))
        pdf_files = pdf_files[:num]
        print(f"Processing first {num} PDFs")
    else:
        print(f"Processing all {len(pdf_files)} PDFs")
    
    # Process all PDF files
    print(f"\nStarting to process {len(pdf_files)} PDF files...")
    print("=" * 80)
    
    successful_files = []
    failed_files = []
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        print(f"Path: {pdf_file}")
        
        try:
            # Process the document
            await rag.process_document_complete(
                file_path=str(pdf_file),
                output_dir=OUTPUT_DIR,
                parse_method="auto",
                display_stats=True
            )
            print(f"✅ Successfully processed: {pdf_file.name}")
            successful_files.append(pdf_file.name)
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error processing {pdf_file.name}: {error_msg[:200]}...")
            failed_files.append((pdf_file.name, error_msg))
            # Continue with next file instead of stopping
            continue
    
    print("\n" + "=" * 80)
    print("PDF Processing Complete!")
    print("=" * 80)
    print(f"✅ Successfully processed: {len(successful_files)} files")
    print(f"❌ Failed to process: {len(failed_files)} files")
    
    if failed_files:
        print("\nFailed files:")
        for filename, _ in failed_files:
            print(f"  - {filename}")
    
    if len(successful_files) == 0:
        print("\n⚠️  No files were successfully processed. Cannot proceed to querying.")
        return
    
    # Interactive query loop
    print("\nYou can now query the processed documents.")
    print("Available query modes: hybrid, local, global, naive")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            # Ask for query mode
            mode = input("Query mode (hybrid/local/global/naive) [hybrid]: ").strip() or "hybrid"
            
            print(f"\nQuerying with mode '{mode}'...")
            result = await rag.aquery(question, mode=mode)
            
            print("\n" + "=" * 80)
            print("ANSWER:")
            print("=" * 80)
            print(result)
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during query: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
