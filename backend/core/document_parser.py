import re
from pathlib import Path
from typing import List, Dict

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

def clean_text(text: str) -> str:
    """
    Cleans raw text by removing non-printable characters and normalizing whitespace.
    """
    if not text:
        return ""
    # Normalize whitespaces to single spaces
    text = re.sub(r"\s+", " ", text)
    # Remove control characters but keep standard punctuation and letters
    text = re.sub(r"[^\x20-\x7E\n\t]", "", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
    """
    Splits text into overlapping chunks of a target character length.
    Ensures that text splits happen at word boundaries (spaces) to preserve meaning.
    """
    chunks = []
    if not text:
        return chunks
    
    text_len = len(text)
    start = 0
    
    while start < text_len:
        # Initial end position
        end = min(start + chunk_size, text_len)
        
        # If we are not at the end of the document, look back to find a word boundary
        if end < text_len:
            # Search for a space in the last 15% of the chunk to split cleanly
            lookback_limit = int(chunk_size * 0.15)
            last_space = text.rfind(" ", end - lookback_limit, end)
            if last_space != -1:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # If we reached the end of the document, we stop
        if end >= text_len:
            break
            
        # Move start position back by overlap
        start = end - overlap
        
        # Guard against zero progress or infinite loops
        if start >= end:
            start = end + 1
            
    return chunks

def extract_text_from_file(file_path: Path) -> str:
    """
    Extracts raw text from a document based on its file extension.
    Supports .txt, .md, and .pdf.
    """
    extension = file_path.suffix.lower()
    
    if extension in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
            
    elif extension == ".pdf":
        if pypdf is None:
            raise ImportError(
                "The 'pypdf' library is required to parse PDF files. "
                "Please install it using: pip install pypdf"
            )
            
        text_parts = []
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    
        return "\n".join(text_parts)
        
    elif extension == ".docx":
        if docx is None:
            raise ImportError(
                "The 'python-docx' library is required to parse DOCX files. "
                "Please install it using: pip install python-docx"
            )
            
        doc = docx.Document(file_path)
        text_parts = []
        
        # Extract paragraph text
        for paragraph in doc.paragraphs:
            val = paragraph.text.strip()
            if val:
                text_parts.append(val)
                
        # Extract table cells cleanly, converting rows to pipe-separated contexts
        for table in doc.tables:
            for row in table.rows:
                row_texts = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    # De-duplicate adjacent identical cells (occurs with merged cells in python-docx)
                    if cell_text and (not row_texts or cell_text != row_texts[-1]):
                        row_texts.append(cell_text)
                if row_texts:
                    text_parts.append(" | ".join(row_texts))
                    
        return "\n".join(text_parts)
        
    else:
        raise ValueError(f"Unsupported file type: {extension}")

def process_document(file_path: Path, chunk_size: int = 600, overlap: int = 150) -> List[str]:
    """
    Parses a document and splits it using the legacy overlapping chunker.
    """
    extension = file_path.suffix.lower()
    
    if extension not in [".docx", ".pdf", ".txt", ".md"]:
        raise ValueError(f"Unsupported file type: {extension}")

    raw_text = extract_text_from_file(file_path)
    cleaned_text = clean_text(raw_text)
    return chunk_text(cleaned_text, chunk_size=chunk_size, overlap=overlap)
