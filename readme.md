# Saga Sage
A RAG designed to learn fiction books.

## Motivation
I like to read long series of fiction books. Sometimes life gets in the way, or the author hasn't finished the series yet, and we're forced to stop reading for a while.
This poses a problem: we may forget important details or story bits during the pause. Some common solutions to this are re-reading the entire series (too time consuming), looking things up on the internet (spoilers!!!) or asking a friend (annoying???). This repo aims to solve that in a different way: having an AI trained on the same books you've read and answer any questions you have of them. "What happened to this character?" "How did they die?" "How can X be Y's father if her mother never took him as a lover?" You're no longer in danger of discovering something you weren't supposed to know yet, or worse, be forced to talk to someone.

## Implementation
I'm starting off with a typical system built using Langchain and OpenAI embeddings stored in a Chroma database. I expect some improvements to come from trying different models for both embeddings and chat, but mostly from how I built the database. Linear fiction series have distinct characteristics from other types of documents, since more recent pieces of context are usually correct when they contradict older ones. Thus, my initial aim is to find a way of biasing the system toward pieces of text relevant to the input question that appear as late as possible in the text.

Another thing to try is having it be capable of extracting "implicit" information that's not directly stated in the text, but that would be relatively easy to discern for a human being. For example, if at 3 separate points in a book we see:
- Character A is an orphan
- Character B had a kid (nobody knows who it is)
- Character B swapped eye color with character C
- Character A has the same eye color as character C

From here we can interpret that A is B's offspring, however a basic RAG would have problems concluding that. Different approaches I might follow are experimenting with context and chunk sizes and some summarizing strategy, where I store embeddings summaries of chapters or key sections along with the ones from raw text.

## Usage
Start by creating a virtual environment
```
python3 -m venv .venv
```
Install dependencies
```
pip install -r requirements.txt
```

Define an environment variable called `OPENAI_API_KEY` with your OpenAI API key (duh), and a `DB_DIR` with the directory where you want to store your embeddings. These can go either in your shell environment or in a .env file.

Put the .epub files you want to extract knowledge from in a "books" directory.

Generate the embeddings database
```
python3 generate_db.py --db_dir /path/to/database/directory
```
Query the database with your question
```
python3 query_db.py --question <your question>
```
