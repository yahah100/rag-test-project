import os
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class QuestionAnswer(BaseModel):
    """Single Q&A pair with metadata"""
    question: str = Field(description="The question that can be answered from the document")
    answer: str = Field(description="The detailed answer to the question")
    source_type: str = Field(description="Whether the answer comes from 'table' or 'text'")
    category: str = Field(description="Question category: 'cost', 'cost_variance', 'delay', 'schedule', 'quality', 'safety', or 'general'")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")
    page_reference: Optional[str] = Field(description="Page number or section where the answer is found", default=None)


class DocumentQAPairs(BaseModel):
    """Collection of Q&A pairs for a document"""
    document_name: str = Field(description="Name of the PDF document")
    qa_pairs: List[QuestionAnswer] = Field(description="List of question-answer pairs")
    total_questions: int = Field(description="Total number of questions generated")


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration settings"""
    # API Settings
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
    MODEL_NAME = "gemini-2.5-pro"  
    
    # Folder Settings
    PDF_FOLDER = "datasets/PMOC_samples"  # Folder containing PDF files
    OUTPUT_FOLDER = "datasets"
    OUTPUT_FILE = "qa_ground_truth.json"
    
    # Generation Settings
    QUESTIONS_PER_DOCUMENT = 15  # Number of questions to generate per document
    TEMPERATURE = 0.7  # Higher = more creative, Lower = more deterministic


# ============================================================================
# Prompt Template
# ============================================================================

EXTRACTION_PROMPT = """You are an expert assistant analyzing construction site monthly reports. Your task is to generate high-quality question-answer pairs that can be used as ground truth for testing a RAG (Retrieval-Augmented Generation) system.

## Instructions:

1. **Carefully read the entire document**, including all text, tables, charts, and data.

2. **Generate {num_questions} diverse questions** covering these priority topics:
   - Project costs and budget information
   - Cost variances (differences between planned vs actual costs)
   - Delays and schedule issues
   - Reasons for delays or setbacks
   - Progress updates
   - Quality and safety metrics
   - Resource allocation
   - Change orders

3. **For each question, provide:**
   - A clear, specific question
   - A detailed, accurate answer based ONLY on information in the document
   - The source type: "table" if the answer comes from tabular data, or "text" if from prose/paragraphs
   - A category: "cost", "cost_variance", "delay", "schedule", "quality", "safety", or "general"
   - A confidence level: "high" (certain), "medium" (somewhat certain), or "low" (uncertain)
   - Page reference if identifiable (optional)

4. **Question Quality Guidelines:**
   - Questions should be specific and answerable
   - Avoid yes/no questions; prefer "what", "how much", "why", "when"
   - Make questions realistic (what a project manager would ask)
   - Ensure questions test both simple fact retrieval and complex reasoning
   - Include questions that require extracting data from tables
   - Vary question complexity (easy, medium, hard)

5. **Important:**
   - Do NOT make up information
   - If data is unclear or missing, mark confidence as "low"
   - Extract exact numbers and dates when available
   - Preserve units (dollars, percentages, days, etc.)

## Examples:

**Table-based question:**
- Question: "What was the total labor cost in March 2024?"
- Answer: "The total labor cost in March 2024 was $245,000."
- Source: "table"
- Category: "cost"

**Text-based question:**
- Question: "What was the primary reason for the 2-week delay in foundation work?"
- Answer: "The foundation work was delayed by 2 weeks due to unexpected soil contamination discovered during excavation, which required additional environmental testing and remediation."
- Source: "text"
- Category: "delay"

Now analyze the provided document and generate the Q&A pairs.
"""


# ============================================================================
# Main Processing Functions
# ============================================================================

def initialize_client(api_key: str) -> genai.Client:
    """Initialize the Google GenAI client"""
    return genai.Client(api_key=api_key)


def get_pdf_files(folder_path: str) -> List[Path]:
    """Get all PDF files from the specified folder"""
    pdf_folder = Path(folder_path)
    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder '{folder_path}' not found. Please create it and add PDF files.")
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{folder_path}'.")
    
    return pdf_files


def process_pdf(client: genai.Client, pdf_path: Path, num_questions: int = 15) -> DocumentQAPairs:
    """Process a single PDF and extract Q&A pairs"""
    print(f"\nProcessing: {pdf_path.name}")
    
    # Read PDF file
    pdf_data = pdf_path.read_bytes()
    
    # Create the prompt
    prompt = EXTRACTION_PROMPT.format(num_questions=num_questions)
    
    # Generate content with structured output
    response = client.models.generate_content(
        model=Config.MODEL_NAME,
        contents=[
            types.Part.from_bytes(
                data=pdf_data,
                mime_type='application/pdf',
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=DocumentQAPairs,
            temperature=Config.TEMPERATURE,
        )
    )
    
    # Parse the response
    result = json.loads(response.text)
    qa_pairs = DocumentQAPairs(**result)
    
    # Update document name
    qa_pairs.document_name = pdf_path.name
    qa_pairs.total_questions = len(qa_pairs.qa_pairs)
    
    print(f"  ✓ Generated {qa_pairs.total_questions} Q&A pairs")
    
    return qa_pairs


def save_results(all_qa_pairs: List[DocumentQAPairs], output_folder: str, output_file: str):
    """Save all Q&A pairs to a JSON file"""
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    output_filepath = output_path / output_file
    
    # Convert to dictionary format
    results = {
        "metadata": {
            "total_documents": len(all_qa_pairs),
            "total_questions": sum(doc.total_questions for doc in all_qa_pairs),
            "model_used": Config.MODEL_NAME,
        },
        "documents": [doc.model_dump() for doc in all_qa_pairs]
    }
    
    # Save to JSON
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_filepath}")
    
    # Generate statistics
    generate_statistics(all_qa_pairs)


def generate_statistics(all_qa_pairs: List[DocumentQAPairs]):
    """Generate and print statistics about the Q&A pairs"""
    total_questions = sum(doc.total_questions for doc in all_qa_pairs)
    
    # Count by source type
    table_count = 0
    text_count = 0
    
    # Count by category
    category_counts = {}
    
    for doc in all_qa_pairs:
        for qa in doc.qa_pairs:
            if qa.source_type == "table":
                table_count += 1
            else:
                text_count += 1
            
            category_counts[qa.category] = category_counts.get(qa.category, 0) + 1
    
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total Documents: {len(all_qa_pairs)}")
    print(f"Total Questions: {total_questions}")
    print(f"\nBy Source Type:")
    print(f"  - Table-based: {table_count} ({table_count/total_questions*100:.1f}%)")
    print(f"  - Text-based:  {text_count} ({text_count/total_questions*100:.1f}%)")
    print(f"\nBy Category:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {category}: {count} ({count/total_questions*100:.1f}%)")
    print("="*60)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    print("="*60)
    print("PDF Q&A Extraction for RAG Ground Truth")
    print("="*60)
    
    # Check API key
    if Config.GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  ERROR: Please set your GEMINI_API_KEY environment variable")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        return
    
    # Initialize client
    print(f"\nInitializing Google AI client (Model: {Config.MODEL_NAME})...")
    client = initialize_client(Config.GEMINI_API_KEY)
    
    # Get PDF files
    try:
        pdf_files = get_pdf_files(Config.PDF_FOLDER)
        print(f"Found {len(pdf_files)} PDF file(s) in '{Config.PDF_FOLDER}'")
    except FileNotFoundError as e:
        print(f"\n⚠️  ERROR: {e}")
        return
    
    # Process each PDF
    all_qa_pairs = []
    for pdf_path in pdf_files:
        try:
            qa_pairs = process_pdf(client, pdf_path, Config.QUESTIONS_PER_DOCUMENT)
            all_qa_pairs.append(qa_pairs)
        except Exception as e:
            print(f"  ✗ Error processing {pdf_path.name}: {e}")
            continue
    
    # Save results
    if all_qa_pairs:
        save_results(all_qa_pairs, Config.OUTPUT_FOLDER, Config.OUTPUT_FILE)
    else:
        print("\n⚠️  No Q&A pairs were generated.")


if __name__ == "__main__":
    main()
