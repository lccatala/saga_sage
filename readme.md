# Saga Sage
A RAG designed to learn fiction books.

## Motivation
I like to read long series of fiction books. Sometimes life gets in the way, or the author hasn't finished the series yet, and we're forced to stop reading for a while.
This poses a problem: we may forget important details or story bits during the pause. Some common solutions to this are re-reading the entire series (too time consuming), looking things up on the internet (spoilers!!!) or asking a friend (annoying???). This repo aims to solve that in a different way: having an AI trained on the same books you've read and answer any questions you have of them. "What happened to this character?" "How did they die?" "How can X be Y's father if her mother never took him as a lover?" You're no longer in danger of discovering something you weren't supposed to know yet, or worse, be forced to talk to someone.

## Implementation
I'm starting off with a typical system built using Langchain and OpenAI embeddings stored in a Chroma database. I expect some improvements to come from trying different models for both embeddings and chat, but mostly from how I built the database. Linear fiction series have distinct characteristics from other types of documents, since more recent pieces of context are usually correct when they contradict older ones. Thus, my initial aim is to find a way of biasing the system toward pieces of text relevant to the input question that appear as late as possible in the text.
