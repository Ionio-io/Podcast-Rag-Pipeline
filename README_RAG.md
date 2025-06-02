# Transcript RAG System

This is a Retrieval-Augmented Generation (RAG) system for querying YouTube video transcripts. The system uses OpenAI's API to create embeddings and generate responses to questions about the content of your transcripts.

## Features

- Upload and process multiple transcript JSON files
- Interactive chat interface for asking questions
- Source tracking to see where answers come from
- Persistent vector store for efficient retrieval
- Conversation memory for context-aware responses

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements_rag.txt
```

2. Set up your OpenAI API key:
   - You can enter it directly in the Streamlit interface
   - Or set it as an environment variable: `OPENAI_API_KEY`

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. In the sidebar:
   - Enter your OpenAI API key
   - Upload your transcript JSON files
   - Click "Process Transcripts" to initialize the RAG system

3. In the main interface:
   - Type your questions in the chat input
   - View responses and their sources
   - Expand the "View Sources" section to see where the information comes from

## File Structure

- `app.py`: Streamlit web interface
- `rag_system.py`: Core RAG system implementation
- `requirements_rag.txt`: Required Python packages
- `chroma_db/`: Directory for storing vector embeddings (created automatically)

## Notes

- The system uses GPT-3.5-turbo for generating responses
- Transcripts are split into chunks of 1000 characters with 200 character overlap
- The system retrieves the 3 most relevant chunks for each question
- All processing is done in memory, with the vector store persisted to disk 