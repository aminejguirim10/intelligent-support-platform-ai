import io
from typing import List, Tuple, Union
import pandas as pd
import pdfplumber
import docx

def parse_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")

def parse_docx(content: bytes) -> str:
    doc = docx.Document(io.BytesIO(content))
    return "\n".join([para.text for para in doc.paragraphs])

def parse_pdf(content: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def parse_csv_to_tickets(content: bytes) -> List[str]:
    """
    Parses a CSV file and returns a list of ticket strings (one per row).
    It tries to find common columns like description, text, body, or ticket.
    """
    df = pd.read_csv(io.BytesIO(content))
    
    # Clean column names (lowercase and strip whitespace)
    columns_map = {col: col.lower().strip() for col in df.columns}
    df_clean = df.rename(columns=columns_map)
    
    # Look for description-like columns
    target_cols = ["description", "text", "body", "comment", "ticket", "content", "issue", "message"]
    found_col = None
    for target in target_cols:
        if target in df_clean.columns:
            # Map back to original column name
            found_col = [orig for orig, clean in columns_map.items() if clean == target][0]
            break
            
    tickets = []
    if found_col:
        # Extract target column values
        tickets = df[found_col].dropna().astype(str).tolist()
    else:
        if len(df.columns) == 1:
            # If only 1 column, use it
            tickets = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            # Otherwise, serialize each row as key-value pairs
            for _, row in df.iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                if row_str.strip():
                    tickets.append(row_str)
                    
    # Clean up whitespace and filter empty tickets
    return [t.strip() for t in tickets if t.strip()]

def extract_tickets_or_text(filename: str, content: bytes) -> Tuple[bool, Union[List[str], str]]:
    """
    Returns (is_list, parsed_content).
    If is_list is True, parsed_content is a List[str] representing already-split CSV rows.
    If is_list is False, parsed_content is a single raw text string (from PDF, DOCX, TXT).
    """
    ext = filename.lower().split('.')[-1]
    if ext == "csv":
        return True, parse_csv_to_tickets(content)
    elif ext == "docx":
        return False, parse_docx(content)
    elif ext == "pdf":
        return False, parse_pdf(content)
    else:
        # Default to txt/plain text
        return False, parse_txt(content)
