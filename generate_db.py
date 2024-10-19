import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def _epub_to_document(path: str) -> list[Document]:
    """
    Extract text content from an epub file.
    
    Args:
        path (str): Path to the EPUB file
        
    Returns:
        dict: Dictionary containing book metadata and content
    """
    loader = UnstructuredEPubLoader(path)
    data = loader.load()
    return data

def _load_books(books_dir: str) -> list:
    print(f"Loading epub files from {books_dir}...")
    books = []
    book_titles = os.listdir(books_dir)
    for book in book_titles:
        book_path = os.path.join(books_dir, book)

        try:
            document = _epub_to_document(book_path)
            books.append(document)
        except Exception as e:
            print(f"Error processing epub:\n{str(e)}")

    return books

def _split_text(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def create_database(books_dir: str) -> None:
    book_list = _load_books(books_dir)
    chunks = []
    for book_documents in book_list:
        new_chunks = _split_text(book_documents)
        chunks.extend(new_chunks)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        print("Could not find an environment variable OPENAI_API_KEY with your OpenAI API key")
        exit()
    db_dir = os.getenv("DB_DIR")
    if db_dir is None:
        print(f"DB_DIR environment variable not defined. Please use it to specify a directory for your vector database")
        exit()

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=db_dir)

    print(f"Stored {len(chunks)} chunks in {db_dir}")


if __name__ == "__main__":
    create_database("books")

