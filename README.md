# RAG Chatbot using LangChain, ChromaDB, Gemini, and Streamlit

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded documents.

## Technologies Used

* Python
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Google Gemini
* Streamlit

## Features

* Document ingestion and chunking
* Vector database creation using ChromaDB
* Semantic search with embeddings
* Answer generation using Gemini
* Interactive Streamlit interface

## Installation

pip install -r requirements.txt

## Configure API Key

Create a .env file:

GOOGLE_API_KEY=YOUR_API_KEY

## Run

python src/ingest.py

streamlit run app.py
