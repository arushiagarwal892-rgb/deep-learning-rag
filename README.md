# Deep Learning PDF RAG

A Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from a deep learning PDF.

## Features

- Loads PDF documents using LangChain.
- Splits documents into smaller text chunks.
- Generates embeddings using Google's Gemini embedding model.
- Stores and retrieves document chunks using ChromaDB.
- Uses a Gemini chat model to generate answers based on retrieved context.

## Tech Stack

- Python
- LangChain
- Google Gemini API
- ChromaDB
- PyPDF

## Setup

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API key:

   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

5. Place your own copy of the PDF in the expected location, if you have permission to use it.
6. Run the application:

   ```bash
   python main.py
   ```

## Notes

- Never commit your real API key.
- The ChromaDB vector store is stored locally and is not included in this repository by default.
- The vector database must be created from the source PDF before running retrieval, unless you provide another supported way to load it.
