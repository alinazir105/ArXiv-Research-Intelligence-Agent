import requests
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from openai import OpenAI

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def download_pdf(url: str) -> bytes:
    """
    Download a PDF file from the given URL and return its content as bytes.
    
    Args:
        url (str): The URL of the PDF file to download.
    
    Returns:
        bytes: The content of the downloaded PDF file.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    return response.content

def extract_text(pdf_bytes: bytes) -> str:
    """
    Extracts clean text from a PDF file provided as bytes, filtering headers/footers and the references section.

    Args:
        pdf_bytes (bytes): The content of the PDF file as bytes.

    Returns:
        str: The extracted clean text from the PDF.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = []

    for page in doc: # loop over pages first
        page_height = page.rect.height    
        page_text = []

        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, block_no, block_type = block

            # skip non-text blocks (images etc)
            if block_type != 0:
                continue

            # skip headers and footers by position
            if y0 < page_height * 0.07 or y1 > page_height *0.93:
                continue

            # skip empty blocks
            if not text.strip():
                continue

            page_text.append(text.strip())

        # check if this page starts the references section — stop here
        if page_text and any(
            page_text[0].strip().startswith(ref)
            for ref in ["References", "Bibliography", "REFERENCES"]
        ):
            break

        full_text.extend(page_text)

    return "\n\n".join(full_text)


def chunk_and_contextualize(text: str, title: str, openai_client) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,       # Maximum number of characters per chunk
        chunk_overlap=300,     # Number of characters shared between adjacent chunks
    )

    chunks = splitter.split_text(text)

    result = []

    for chunk in chunks:
        # generate context for this specific chunk
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": f"""Document title: {title}

                        Here is a chunk from this research paper:
                        
                        <chunk>
                        {chunk}
                        </chunk>

                        Give a short succinct context (50-100 tokens) to situate this chunk within 
                        the overall document for improving search retrieval. 
                        Answer only with the context, nothing else."""
                }
            ]
        )

        context = response.choices[0].message.content.strip()

        # prepend context to chunk — this is contextual retrieval
        contextual_text = f"{context}\n\n{chunk}"

        result.append({
            "text": contextual_text,
            "title": title,
            "source_type": "full_paper"
        })

    return result

