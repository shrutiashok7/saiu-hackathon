import chromadb
import requests
import json
import os
from pypdf import PdfReader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
# Path to the PDF file you want to process
# You can change this to a default path or leave it to prompt the user
PDF_FILE_PATH = "" 

# Ollama API configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text:latest"

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_embeddings"

# Text chunking configuration
CHUNK_SIZE = 1000  # Size of each text chunk in characters
CHUNK_OVERLAP = 200 # Number of characters to overlap between chunks

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from all pages of a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The concatenated text from all pages.
    """
    print(f"Reading PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in tqdm(reader.pages, desc="Extracting pages"):
            full_text += page.extract_text() or ""
        print(f"Successfully extracted {len(full_text)} characters from the PDF.")
        return full_text
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None

def chunk_text(text, chunk_size, chunk_overlap):
    """
    Splits a long text into smaller, overlapping chunks.

    Args:
        text (str): The input text.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []
        
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
    
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

def get_ollama_embedding(text_chunk):
    """
    Generates an embedding for a given text chunk using the Ollama API.

    Args:
        text_chunk (str): The text to embed.

    Returns:
        list[float]: The generated embedding vector, or None if an error occurs.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": text_chunk
        }
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status() # Raises an HTTPError for bad responses
        response_json = response.json()
        return response_json.get("embedding")
    except requests.exceptions.RequestException as e:
        # Don't print for every failure in sequential mode, just return None
        return None

def store_chunks_in_chromadb(chunks, collection):
    """
    Generates embeddings for a list of text chunks sequentially and stores them in ChromaDB.
    
    Args:
        chunks (list[str]): The list of text chunks.
        collection: The ChromaDB collection object.
    """
    if not chunks:
        print("No chunks to store.")
        return
        
    print(f"Storing {len(chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}'...")
    
    # Prepare lists to hold all data for a single batch add operation
    ids_to_add = []
    embeddings_to_add = []
    documents_to_add = []

    # Process chunks sequentially to avoid overloading the Ollama server
    for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings")):
        embedding = get_ollama_embedding(chunk)
        if embedding:
            ids_to_add.append(str(i))
            embeddings_to_add.append(embedding)
            documents_to_add.append(chunk)
        else:
            print(f"\nWarning: Could not generate embedding for chunk {i}. Skipping.")

    # Add all collected data to ChromaDB in a single batch
    if documents_to_add:
        collection.add(
            ids=ids_to_add,
            embeddings=embeddings_to_add,
            documents=documents_to_add
        )
    
    count = collection.count()
    print(f"\nSuccessfully stored {count} embeddings in the collection.")


def main():
    """
    Main function to run the PDF processing and embedding pipeline.
    """
    # 1. Get PDF Path
    pdf_path = PDF_FILE_PATH
    if not pdf_path:
        pdf_path = input("Please enter the full path to your PDF file: ")

    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'")
        return

    # 2. Extract Text
    document_text = extract_text_from_pdf(pdf_path)
    if not document_text:
        return

    # 3. Chunk Text
    text_chunks = chunk_text(document_text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not text_chunks:
        print("Could not generate any text chunks.")
        return
        
    # 4. Initialize ChromaDB
    print(f"Initializing ChromaDB client at '{CHROMA_DB_PATH}'...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Failed to initialize ChromaDB: {e}")
        return

    # 5. Generate and Store Embeddings
    store_chunks_in_chromadb(text_chunks, collection)
    
    print("\n--- Process Complete ---")
    print(f"PDF processed: {os.path.basename(pdf_path)}")
    print(f"Embeddings stored in ChromaDB collection: '{COLLECTION_NAME}'")
    print(f"Database path: '{CHROMA_DB_PATH}'")

if __name__ == "__main__":
    main()

