import os
import sys
import argparse
import operator

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document

def search_by_position_and_similarity(vector_store: Chroma, question_text: str, k: int = 3) -> list[tuple[Document, float]]:
    results = vector_store.similarity_search_with_relevance_scores(question_text, k=k)
    # Sort based on the start_index in the Document's metadata
    sorted_results = sorted(results, key=lambda x: x[0].metadata.get("start_index", 0), reverse=True)
    return sorted_results
    # results_data: list[tuple] = []
    # results = vector_store.similarity_search_with_relevance_scores(question_text, k=k)
    # for doc, score in results:
    #     position: int = doc.metadata.get("start_index", 0)
    #     source: str = doc.metadata.get("source", "Unknown")
    #     result_doc: Document = Document(doc.page_content, score=score, source=source)
    #     results_data.append(result_doc)
    #
    # sorted_results = sorted(results_data, key=operator.__getitem__, reverse=True) 
    # return sorted_results

def ask_question(
    question_text: str, 
    db_root_dir: str, 
    min_similarity: float = 0.7,
    max_context_pieces: int = 5
) -> dict[str, str | list[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        sys.exit("Could not find an OpenAI API key")

    embedding_function = OpenAIEmbeddings()
    book_directories = os.listdir(db_root_dir)
    results = []
    for book_dir in book_directories:
        print("1")
        book_db_path = os.path.join(db_root_dir, book_dir)
        print(book_db_path)
        try:
            db = Chroma(persist_directory=book_db_path, embedding_function=embedding_function)
        except Exception as e:
            sys.exit(str(e))
        print("3")
        book_results = search_by_position_and_similarity(db, question_text, k=3)
        results.extend(book_results)
        if len(results) >= max_context_pieces:
            results = results[:max_context_pieces]
            break
    if len(results) == 0 or results[0][1] < min_similarity:
        sys.exit(f"Could not find relevant results")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    PROMPT_TEMPLATE = """
    Answer the question based on the following context:
    {context}

    ---

    Answer the question based on the above context: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question_text)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    sources = [doc.metadata.get("source", None) for doc, _ in results]

    return {"answer":response_text, "sources":sources}

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ask a question using the generated database from the books you provided")
    parser.add_argument("--db_dir", type=str, default="chroma", help="Directory path for the generated database")
    parser.add_argument("--question", type=str, default="What is the River?", help="Question to ask the system related to the books you provided")
    args = parser.parse_args()


    answer = ask_question(args.question, args.db_dir)
    print(answer["answer"])
