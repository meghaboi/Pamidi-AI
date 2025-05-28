import uuid
import os
import logging
import json
import time
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse, Response # Added Response
from pydantic import BaseModel

# Assuming backend_utils.py, pipeline_utils.py, subject_configs.py, enums.py are in the same directory (backend/)
from .backend_utils import save_uploaded_file_async as save_uploaded_file # Use async version
from .backend_utils import check_api_keys, text_to_speech, is_greeting # Added text_to_speech and is_greeting
from .pipeline_utils import initialize_pipeline
from .subject_configs import (
    DEFAULT_EMBEDDING_MODEL, 
    DEFAULT_VECTOR_STORE, 
    DEFAULT_RERANKER_MODEL, 
    DEFAULT_LLM_MODEL, 
    DEFAULT_CHUNKING_STRATEGY, 
    DEFAULT_HYBRID_ALPHA, 
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_TOP_K
)
from .enums import EmbeddingModelType, VectorStoreType, RerankerModelType, LLMModelType, ChunkingStrategyType

app = FastAPI()
router = APIRouter()

# Global storage for active RAG pipelines
active_pipelines: dict = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    session_id: str
    query: str

class TTSRequest(BaseModel):
    text: str

class GreetingRequest(BaseModel):
    query: str

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Received file upload request for: {file.filename}")

    saved_file_path = await save_uploaded_file(file)
    if saved_file_path is None:
        logger.error("Failed to save uploaded file.")
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    logger.info(f"File saved to: {saved_file_path}")

    # Check for API keys using default models.
    # Note: check_api_keys expects enum members, not their string values.
    missing_keys = check_api_keys(
        embedding_model_enum=DEFAULT_EMBEDDING_MODEL, # Direct enum member
        vector_store_enum=DEFAULT_VECTOR_STORE,       # Direct enum member
        reranker_enum=DEFAULT_RERANKER_MODEL,         # Direct enum member
        llm_enum=DEFAULT_LLM_MODEL,                   # Direct enum member
        return_missing=True
    )

    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        # Clean up the saved file if keys are missing
        if os.path.exists(saved_file_path):
            os.remove(saved_file_path)
        raise HTTPException(status_code=400, detail=f"Missing API keys: {', '.join(missing_keys)}")

    logger.info("API keys check passed.")

    try:
        logger.info("Initializing RAG pipeline...")
        # Ensure that initialize_pipeline is called with enum members where expected
        pipeline = initialize_pipeline(
            file_path=saved_file_path,
            embedding_model_enum=DEFAULT_EMBEDDING_MODEL,
            vector_store_enum=DEFAULT_VECTOR_STORE,
            reranker_enum=DEFAULT_RERANKER_MODEL,
            llm_enum=DEFAULT_LLM_MODEL,
            chunking_strategy_enum=DEFAULT_CHUNKING_STRATEGY, # This was missing as a default in subject_configs, using the direct default
            hybrid_alpha=DEFAULT_HYBRID_ALPHA,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            top_k=DEFAULT_TOP_K,
            evaluation_mode=False # Typically False for operational use
        )

        if pipeline is None:
            logger.error("Failed to initialize RAG pipeline. initialize_pipeline returned None.")
            # Clean up the saved file
            if os.path.exists(saved_file_path):
                os.remove(saved_file_path)
            raise HTTPException(status_code=500, detail="Failed to initialize RAG pipeline.")
        
        logger.info("RAG pipeline initialized successfully.")
        
        session_id = str(uuid.uuid4())
        active_pipelines[session_id] = {
            "pipeline": pipeline, 
            "filename": file.filename,
            "saved_file_path": saved_file_path # Store for potential cleanup or later use
        }
        logger.info(f"Pipeline stored with session_id: {session_id}")

        return JSONResponse(
            content={"message": "Pipeline initialized", "session_id": session_id, "filename": file.filename},
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error during pipeline initialization: {str(e)}", exc_info=True)
        # Clean up the saved file in case of error
        if os.path.exists(saved_file_path):
            os.remove(saved_file_path)
        raise HTTPException(status_code=500, detail=f"Error during pipeline initialization: {str(e)}")

@router.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"Received chat request for session_id: {request.session_id}")
    pipeline_data = active_pipelines.get(request.session_id)

    if not pipeline_data or "pipeline" not in pipeline_data:
        logger.warning(f"Invalid session ID or pipeline not found for session_id: {request.session_id}")
        raise HTTPException(status_code=404, detail="Invalid session ID or pipeline not found. Please upload a file first.")

    pipeline = pipeline_data["pipeline"]
    if pipeline is None: # Should be caught by above, but as a safeguard
        logger.error(f"Pipeline object is None for session_id: {request.session_id}")
        raise HTTPException(status_code=404, detail="Pipeline not properly initialized. Please re-upload the file.")


    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.time()
        full_response = ""
        contexts = [] # Initialize contexts in a broader scope
        
        try:
            logger.info(f"Streaming response for query: {request.query} using session_id: {request.session_id}")
            async for chunk in pipeline.stream_run(request.query): # Assuming stream_run is async
                if chunk is not None:
                    full_response += chunk
                    json_chunk = json.dumps({"type": "chunk", "data": chunk})
                    yield f"data: {json_chunk}\n\n"
        except Exception as e:
            logger.error(f"Error during response streaming for session_id {request.session_id}: {str(e)}", exc_info=True)
            error_json = json.dumps({"type": "error", "data": f"Error streaming response: {str(e)}"})
            yield f"data: {error_json}\n\n"
            # Optionally, re-raise or handle differently
            # For now, we will proceed to finally to send any accumulated data and contexts if possible
        finally:
            elapsed_time = time.time() - start_time
            try:
                # Retrieve contexts even if streaming had issues, if pipeline is available
                contexts = pipeline.retrieve_context(request.query)
                logger.info(f"Retrieved {len(contexts)} contexts for session_id: {request.session_id}")
            except Exception as e:
                logger.error(f"Error retrieving contexts for session_id {request.session_id}: {str(e)}", exc_info=True)
                contexts = ["Error retrieving contexts."]


            json_final = json.dumps({
                "type": "result",
                "full_response": full_response,
                "contexts": contexts,
                "elapsed_time": round(elapsed_time, 2)
            })
            yield f"data: {json_final}\n\n"
            logger.info(f"Finished streaming for session_id: {request.session_id}, total time: {elapsed_time:.2f}s")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/tts")
async def generate_speech(request: TTSRequest):
    logger.info(f"Received TTS request for text: '{request.text[:50]}...'") # Log first 50 chars
    
    if not request.text.strip():
        logger.warning("TTS request with empty text received.")
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Directly calling the synchronous text_to_speech function.
    # FastAPI will run this in an external thread pool.
    audio_bytes = text_to_speech(request.text)

    if audio_bytes is None:
        logger.error("Text-to-speech generation failed or produced no audio.")
        raise HTTPException(status_code=500, detail="Text-to-speech generation failed or produced no audio.")
    
    logger.info(f"TTS audio generated successfully, size: {len(audio_bytes)} bytes.")
    return Response(content=audio_bytes, media_type="audio/mpeg")

@router.post("/check_greeting")
async def check_greeting_endpoint(request: GreetingRequest):
    logger.info(f"Received greeting check request for query: '{request.query[:50]}...'")

    if not request.query.strip():
        logger.warning("Greeting check request with empty query received.")
        # Consistent with how TTS handles empty text, though a greeting is unlikely to be empty.
        # Alternatively, could return {"is_greeting": False, "response": ""} directly.
        raise HTTPException(status_code=400, detail="Input query cannot be empty.")

    # Directly calling the synchronous is_greeting function.
    # FastAPI will run this in an external thread pool if it's IO-bound.
    try:
        is_greet, greeting_response_text = is_greeting(request.query)
        logger.info(f"Greeting check result: is_greeting={is_greet}, response='{greeting_response_text}'")
        return {"is_greeting": is_greet, "response": greeting_response_text}
    except Exception as e:
        # This is a general catch-all. Specific exceptions from Anthropic client (e.g., API key issues)
        # might be handled more gracefully by is_greeting itself, or here if needed.
        logger.error(f"Error during greeting check for query '{request.query[:50]}...': {str(e)}", exc_info=True)
        # If is_greeting is robust and returns (False, "") on its own errors, this might not be strictly needed
        # unless there are other FastAPI-level or unexpected errors.
        raise HTTPException(status_code=500, detail=f"Error processing greeting check: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API"}

app.include_router(router, prefix="/api")

# Example of how to run for local testing (optional)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
