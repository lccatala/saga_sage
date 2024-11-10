import os
import sys
import shutil
import argparse

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import ebooklib
from ebooklib import epub
import xml.etree.ElementTree as ET
from datetime import datetime


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

def _get_epub_date(book_path: str) -> datetime:
    book = epub.read_epub(book_path)
    # Try all field names where the date might be stored
    date_fields = [ 
        'DC:date',
        'dcterms:created',
        'dcterms:modified',
        'dcterms:issued',
        'opf:publication-date'
    ]
    for field in date_fields:
        date = book.get_metadata("DC", field)
        if date:
            try:
                date_str = date[0][0]
                for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y%m%d', '%Y'):
                    try:
                        return datetime.strptime(date_str[:10], fmt)
                    except ValueError:
                        continue
            except (IndexError, ValueError):
                continue

    # If we couldn't find it, parse the OPF file
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        root = ET.fromstring(item.get_content())
        for meta in root.findall(".//{*}meta"):
            if "date" in (meta.get("name", "") + meta.get("property", "")).lower():
                try:
                    date_str = meta.get("content", "")
                    return datetime.strptime(date_str[:10], "%Y-%m-%d")
                except ValueError:
                    continue

    # If we couldn't find it either, return the file's creation date
    creation_time = os.path.getctime(book_path)
    creation_date = datetime.fromtimestamp(creation_time)
    return creation_date


def _load_books(books_dir: str) -> list:
    print(f"Loading epub files from {books_dir}...")
    books = []
    creation_dates = []
    book_titles = os.listdir(books_dir)
    for book in book_titles:
        book_path = os.path.join(books_dir, book)

        try:
            document = _epub_to_document(book_path)
            books.append(document)

            book_date = _get_epub_date(book_path)
            creation_dates.append(book_date)
        except Exception as e:
            print(f"Error processing epub:\n{str(e)}")

    # Sort the books by their creation dates
    books = [b for b, _ in sorted(zip(books, creation_dates), key=lambda pair: pair[1])]

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

def create_database(books_dir: str, db_dir: str) -> None:
    book_list = _load_books(books_dir)
    if len(book_list) == 0:
        sys.exit("Error: you did not provide any books")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        sys.exit("Could not find an environment variable OPENAI_API_KEY with your OpenAI API key")

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        os.mkdir(db_dir)

    total_chunks = 0
    for i, book_documents in enumerate(book_list):
        chunks = _split_text(book_documents)
        book_db_dir = os.path.join(db_dir, str(i))
        Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=book_db_dir)

        print(f"Stored {len(chunks)} chunks in {book_db_dir}")
        total_chunks += len(chunks)
    print(f"Stored a total of {total_chunks} in {db_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings database")
    parser.add_argument("--db_dir", type=str, default="chroma", help="Directory path for the database to generate")
    args = parser.parse_args()

    load_dotenv()
    create_database("books", args.db_dir)

