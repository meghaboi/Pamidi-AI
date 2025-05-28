# RAG System with FastAPI Backend and JavaScript Frontend

## Overview

This project implements a Retrieval Augmented Generation (RAG) system that allows users to upload textbook documents (PDF, DOCX, TXT) and interact with them through a chat interface. The backend is built using FastAPI (Python), and the frontend is a single-page application using plain HTML, CSS, and JavaScript. The system provides streamed responses, text-to-speech (TTS) for answers, and displays the context snippets used for generation.

## Features

*   **File Upload:** Upload textbook documents in PDF, DOCX, or TXT format.
*   **Chat Interface:** Ask questions about the uploaded document.
*   **Streaming Responses:** Receive answers from the Language Model (LLM) in real-time.
*   **Text-to-Speech (TTS):** Listen to the assistant's responses.
*   **Context Display:** View the relevant text snippets from the document that were used to generate the answer.
*   **Greeting Detection:** Handles simple greetings gracefully before engaging the RAG pipeline.
*   **Modern UI:** Clean and responsive user interface.

## Architecture

*   **Backend:**
    *   Framework: FastAPI (Python)
    *   RAG Pipeline: Custom implementation for document indexing, retrieval, and generation.
    *   Core Logic: Located in the `backend/` directory.
    *   Dependencies: Managed by `backend/requirements.txt`.
    *   Uploaded files are temporarily stored in `backend/temp_files/`.
*   **Frontend:**
    *   Technology: HTML, CSS, JavaScript
    *   Location: Served from the `static/` directory.
    *   No build step required.

## Installation / Setup

### Prerequisites

*   Python 3.8 or newer.
*   `pip` for installing Python packages.
*   Access to the internet for downloading models/dependencies and for API calls.

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### API Keys

The application requires API keys for various services (LLMs, embedding models, TTS, etc.) to function correctly. These keys must be set as environment variables. For the default configuration, the following keys are essential:

*   `OPENAI_API_KEY`: Required for Text-to-Speech (TTS) functionality.
*   `MISTRAL_API_KEY`: Required for the default embedding model.
*   `ANTHROPIC_API_KEY`: Required for the default Language Model (Claude) and for greeting detection.
*   `VOYAGE_API_KEY`: Required for the default reranker model.

If you intend to use other models (e.g., Cohere, Gemini), you will also need to set their respective API keys:
*   `COHERE_API_KEY`
*   `GEMINI_API_KEY`

Set these environment variables in your shell or `.env` file (if you use a library like `python-dotenv`, ensure it's loaded appropriately by your environment). For example:
```bash
export OPENAI_API_KEY="your_openai_key"
export MISTRAL_API_KEY="your_mistral_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export VOYAGE_API_KEY="your_voyage_key"
```

### Frontend Setup

No specific build step is required for the frontend. The static files (`index.html`, `style.css`, `script.js`) are served directly by the FastAPI backend.

## Running the Application

1.  **Start the Backend Server:**
    From the **root directory** of the project, run:
    ```bash
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Enables auto-reloading when code changes (useful for development).
    *   `--host 0.0.0.0`: Makes the server accessible from your local network.
    *   `--port 8000`: Runs the server on port 8000.

2.  **Access the Frontend:**
    Open your web browser and navigate to:
    [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

## Usage Instructions

1.  **Upload a Document:**
    *   Click the "Choose File" button in the "Upload Textbook" section.
    *   Select a supported file (PDF, DOCX, TXT).
    *   Click the "Upload & Process" button.
    *   Wait for the status message to indicate that the file has been processed and the session is ready.
2.  **Chat:**
    *   Once a document is processed, the chat input area will be enabled.
    *   Type your question into the textarea and click "Send" or press Enter.
    *   The assistant's response will be streamed into the chatbox.
3.  **Text-to-Speech (TTS):**
    *   For assistant messages, click the "Hear Response" button to listen to the audio.
4.  **View Contexts:**
    *   Check the "Show Contexts for Assistant Messages" checkbox to view the text snippets retrieved from the document that were used to generate the response. These will appear below the assistant's message.

## Backend Details & Extending (Optional)

The backend logic is primarily contained within the `backend/` directory:

*   `main.py`: Defines the FastAPI application, API endpoints (`/upload`, `/chat`, `/tts`, `/check_greeting`), and manages active RAG pipelines.
*   `pipeline_utils.py`: Contains functions for initializing and running RAG pipeline configurations.
*   `rag_pipeline.py`: Defines the `RAGPipeline` class, chunking strategies, and hybrid search logic.
*   `embedding_models.py`, `llm_models.py`, `rerankers.py`, `vector_stores.py`: Implementations and factories for different model types and vector stores. These can be extended to support new models or services.
*   `backend_utils.py`: Contains utility functions for file handling, API key checks, TTS, and greeting detection.
*   `enums.py`: Defines various Enum types used throughout the backend for model selection, etc.
*   `subject_configs.py`: Defines default model configurations (though subject-specific configurations are currently commented out).

To extend the system (e.g., add a new embedding model):
1.  Implement the new model class in the relevant file (e.g., `backend/embedding_models.py`), ensuring it adheres to the base abstract class.
2.  Add the new model type to the corresponding Enum in `backend/enums.py`.
3.  Update the factory function in the model's file to include the new type.
4.  If the new model requires an API key, update `backend/backend_utils.py`'s `check_api_keys` function and the API Keys section in this README.

## Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── backend_utils.py    # Backend utility functions (file saving, API checks, TTS, greeting)
│   ├── embedding_models.py # Embedding model implementations
│   ├── enums.py            # Enum definitions for models, etc.
│   ├── evaluator.py        # RAG evaluation logic
│   ├── llm_models.py       # LLM implementations
│   ├── main.py             # FastAPI application
│   ├── pipeline_utils.py   # Utilities for RAG pipeline setup
│   ├── rag_pipeline.py     # Core RAG pipeline logic and chunking strategies
│   ├── rerankers.py        # Reranker model implementations
│   ├── requirements.txt    # Backend Python dependencies
│   ├── subject_configs.py  # Default model configurations
│   ├── temp_files/         # Directory for storing uploaded files (auto-created)
│   └── vector_stores.py    # Vector store implementations
├── static/
│   ├── index.html          # Frontend HTML
│   ├── script.js           # Frontend JavaScript
│   └── style.css           # Frontend CSS
├── README.MD               # This file
├── utils.py                # Original utility file (mostly refactored into backend/)
└── enums.py                # Original enums file (refactored into backend/)
```
