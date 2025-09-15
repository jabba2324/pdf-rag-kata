import os
import fitz  # PyMuPDF

PDF_FOLDER = "universal_raw_data"  # folder with PDFs

# ----------- Step 1: Extract text from PDFs -----------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    
    # Create text folder if it doesn't exist
    os.makedirs("text", exist_ok=True)
    
    # Get PDF filename without extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text.strip(), "page": page_num + 1})
            
            # Write page to text file
            text_filename = f"{pdf_name}_page_{page_num + 1}.txt"
            text_filepath = os.path.join("text", text_filename)
            with open(text_filepath, "w", encoding="utf-8") as f:
                f.write(text.strip())
    
    return pages


for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)

        pages = extract_text_from_pdf(pdf_path)
