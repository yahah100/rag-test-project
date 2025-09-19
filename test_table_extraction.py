#!/usr/bin/env python3
"""
Test script to verify table extraction functionality.

This script demonstrates the enhanced PDF processing with table extraction
by showing what gets extracted from PDF files.
"""

import logging
from pathlib import Path
from ask_rag import ImprovedPDFRAG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_table_extraction():
    """
    Test the table extraction functionality by loading a few PDFs
    and showing the extracted content.
    """
    print("🧪 Testing Enhanced PDF Table Extraction")
    print("=" * 50)
    
    # Initialize the RAG system (without full setup)
    rag_system = ImprovedPDFRAG(
        pdf_folder="datasets",
        ollama_model="gemma3:1b",  # We won't use this for testing
        embedding_model="google/embeddinggemma-300m"  # We won't use this for testing
    )
    
    try:
        # Load PDFs with table extraction
        print("\n📚 Loading PDFs with table extraction...")
        documents = rag_system.load_pdfs()
        
        if not documents:
            print("❌ No documents loaded. Make sure you have PDF files in the 'datasets' folder.")
            return
        
        print(f"\n📊 Results Summary:")
        print(f"  • Total documents: {len(documents)}")
        
        total_tables = 0
        enhanced_count = 0
        
        # Show details for each document
        print(f"\n📄 Document Details:")
        for i, doc in enumerate(documents, 1):
            filename = doc.metadata.get("filename", "Unknown")
            pages = doc.metadata.get("total_pages", 0)
            tables = doc.metadata.get("tables_extracted", 0)
            enhanced = doc.metadata.get("enhanced_extraction", False)
            
            total_tables += tables
            if enhanced:
                enhanced_count += 1
            
            extraction_type = "📊 Enhanced" if enhanced else "📝 Basic"
            print(f"  {i}. {filename}")
            print(f"     {extraction_type} | {pages} pages | {tables} tables")
            
            # Show a preview of the content (first 300 chars)
            content_preview = doc.page_content[:300].replace('\n', ' ').strip()
            if len(doc.page_content) > 300:
                content_preview += "..."
            print(f"     Preview: {content_preview}")
            print()
        
        # Summary statistics
        print(f"📈 Extraction Statistics:")
        print(f"  • Documents with enhanced extraction: {enhanced_count}/{len(documents)}")
        print(f"  • Total tables extracted: {total_tables}")
        print(f"  • Average tables per document: {total_tables/len(documents):.1f}")
        
        # Show detailed content for first document with tables
        table_doc = next((doc for doc in documents if doc.metadata.get("tables_extracted", 0) > 0), None)
        if table_doc:
            print(f"\n🔍 Sample Table Content from '{table_doc.metadata.get('filename')}':")
            print("-" * 60)
            
            # Find and show table sections
            content_lines = table_doc.page_content.split('\n')
            in_table = False
            table_lines = []
            
            for line in content_lines:
                if line.strip().startswith('[TABLE'):
                    in_table = True
                    table_lines = [line]
                elif in_table:
                    table_lines.append(line)
                    if not line.strip():  # Empty line marks end of table
                        # Show this table
                        for table_line in table_lines:
                            print(table_line)
                        print()
                        in_table = False
                        break  # Just show first table
        
        print("✅ Table extraction test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        logger.error(f"Test error: {str(e)}")

if __name__ == "__main__":
    test_table_extraction()
