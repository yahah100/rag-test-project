from pathlib import Path
import logging 

import pypdf
from langchain_core.documents import Document

class PDFReader:

    def __init__(self) -> None:
        pass

    def read_pdf(self, pdf_file: Path) -> Document:
        logging.info(f"📖 Processing: {pdf_file.name}")
        
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
                logging.info(f"✅ Extracted {len(pdf_reader.pages)} pages from {pdf_file.name}")
                return doc
            else:
                logging.warning(f"⚠️ No text found in {pdf_file.name}")
                raise ValueError(f"No text found in {pdf_file.name}")
                    
