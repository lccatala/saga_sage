import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def ask_question(question_text: str, min_similarity: float = 0.7) -> dict[str, str | list[str]]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        print("Could not find an OpenAI API key")
        exit()
    db_dir = os.getenv("DB_DIR")
    if db_dir is None:
        print(f"DB_DIR environment variable not defined. Please specify a directory for your vector database")
        exit()

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=db_dir, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(question_text, k=5)
    if len(results) == 0 or results[0][1] < min_similarity:
        print(f"Could not find relevant results")
        exit()

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
    sources = [doc.metadata.get("source", None) for doc, score in results]

    return {"answer":response_text, "sources":sources}

if __name__ == "__main__":
    question = "What is the river?"
    answer = ask_question(question)
    print(answer)
