import os
import tempfile
import logging
from fastapi import UploadFile
from typing import Optional, List
from pathlib import Path

# Assuming enums.py is in the same directory
from .enums import EmbeddingModelType, LLMModelType, RerankerModelType, VectorStoreType 

TEMP_FILES_DIR = Path("backend/temp_files")

def save_uploaded_file(file: UploadFile) -> Optional[str]:
    """Save uploaded file to a temporary location and return the path"""
    try:
        TEMP_FILES_DIR.mkdir(parents=True, exist_ok=True)
        file_suffix = Path(file.filename).suffix if '.' in file.filename else '.txt'
        # Use await file.read() for FastAPI's UploadFile
        file_content = file.file.read() # Corrected: Use file.file.read() for synchronous read or await file.read() if in async context
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, dir=TEMP_FILES_DIR, mode='wb') as temp:
            temp.write(file_content)
            logging.info(f"Saved uploaded file '{file.filename}' to temporary path: {temp.name}")
            return temp.name
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        return None

async def save_uploaded_file_async(file: UploadFile) -> Optional[str]:
    """Save uploaded file to a temporary location and return the path (async version)"""
    try:
        TEMP_FILES_DIR.mkdir(parents=True, exist_ok=True)
        file_suffix = Path(file.filename).suffix if '.' in file.filename else '.txt'
        file_content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, dir=TEMP_FILES_DIR, mode='wb') as temp:
            temp.write(file_content)
            logging.info(f"Saved uploaded file '{file.filename}' to temporary path: {temp.name}")
            return temp.name
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        return None

def check_api_keys(embedding_model_enum: EmbeddingModelType, 
                     vector_store_enum: VectorStoreType, # Added vector_store_enum as it's used
                     reranker_enum: RerankerModelType, 
                     llm_enum: LLMModelType, 
                     return_missing: bool = False) -> Optional[List[str]]:
    """Check if required API keys are available in environment.
    If return_missing is True, returns a list of missing key names.
    Otherwise, returns None (as st.session_state is removed).
    """
    # api_keys_status = {} # Not needed without session_state
    missing_keys_list = []

    # Determine required keys based on selections
    openai_needed = (embedding_model_enum == EmbeddingModelType.OPENAI or
                     llm_enum in [LLMModelType.OPENAI_GPT35, LLMModelType.OPENAI_GPT4] or
                     True) # OpenAI TTS always needs it (though TTS is not directly in this function's scope)
    cohere_needed = (embedding_model_enum == EmbeddingModelType.COHERE or
                     reranker_enum in [RerankerModelType.COHERE_V2, RerankerModelType.COHERE_V3, RerankerModelType.COHERE_MULTILINGUAL])
    gemini_needed = (embedding_model_enum == EmbeddingModelType.GEMINI or
                     llm_enum == LLMModelType.GEMINI)
    anthropic_needed = (llm_enum in [LLMModelType.CLAUDE_3_OPUS, LLMModelType.CLAUDE_37_SONNET]) # Corrected: CLAUDE_37_SONNET
    mistral_needed = (embedding_model_enum == EmbeddingModelType.MISTRAL or
                      llm_enum in [LLMModelType.MISTRAL_LARGE, LLMModelType.MISTRAL_MEDIUM, LLMModelType.MISTRAL_SMALL])
    voyage_needed = (embedding_model_enum == EmbeddingModelType.VOYAGE or
                     reranker_enum in [RerankerModelType.VOYAGE, RerankerModelType.VOYAGE_2])

    # Check and record status
    if openai_needed:
        key_name = "OpenAI API Key"
        is_available = bool(os.getenv("OPENAI_API_KEY"))
        # api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if cohere_needed:
        key_name = "Cohere API Key"
        is_available = bool(os.getenv("COHERE_API_KEY"))
        # api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if gemini_needed:
        key_name = "Gemini API Key"
        is_available = bool(os.getenv("GEMINI_API_KEY"))
        # api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if anthropic_needed:
        key_name = "Anthropic API Key"
        is_available = bool(os.getenv("ANTHROPIC_API_KEY"))
        # api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if mistral_needed:
        key_name = "Mistral API Key"
        is_available = bool(os.getenv("MISTRAL_API_KEY"))
        # api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if voyage_needed:
        key_name = "Voyage AI API Key"
        is_available = bool(os.getenv("VOYAGE_API_KEY"))
        # api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if return_missing:
        return missing_keys_list
    else:
        # st.session_state.api_key_status = api_keys_status # Removed
        return None # Or return [] if a list is always expected by caller when not return_missing

# Copied from root utils.py and modified
import re
import httpx
from openai import OpenAI

def text_to_speech(text: str) -> Optional[bytes]:
    """Generates speech from text using OpenAI TTS and returns audio bytes."""
    if not text or not isinstance(text, str):
        logging.warning("TTS skipped: Input text is empty or not a string.")
        return None

    cleaned_text = re.sub(r'[#*]', '', text) 
    cleaned_text = re.sub(r'http[s]?://\S+', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    if not cleaned_text:
        logging.warning("TTS skipped: Text is empty after cleaning.")
        return None

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OpenAI API Key not found. Cannot generate audio.")
        return None

    try:
        # Ensure OpenAI client is properly initialized
        # httpx.Timeout settings can be adjusted as needed
        client = OpenAI(timeout=httpx.Timeout(45.0, connect=10.0)) 
        selected_voice = "fable"  # Default voice
        selected_model = "tts-1"  # Default model

        logging.info(f"Requesting OpenAI TTS: voice='{selected_voice}', model='{selected_model}', text length (cleaned): {len(cleaned_text)}")

        response = client.audio.speech.create(
            model=selected_model,
            voice=selected_voice,
            input=cleaned_text,
            response_format="mp3"  # Ensure response format is mp3
        )

        audio_bytes = response.read() # Reads the response content as bytes
        logging.info(f"OpenAI TTS audio generated successfully ({len(audio_bytes)} bytes).")
        return audio_bytes

    except ImportError:
        logging.error("OpenAI library not installed. Cannot generate audio.")
        return None
    except Exception as e:
        logging.error(f"Error generating OpenAI TTS audio: {e}", exc_info=True)
        return None

# Copied from root utils.py and modified
from anthropic import Anthropic
import random

def is_greeting(query: str) -> tuple[bool, str]:
    """Detect if the query is a greeting using Anthropic's function calling and get the response."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        logging.error("ANTHROPIC_API_KEY not found. Cannot perform greeting detection.")
        return (False, "") # Return False, empty string if key is missing

    try:
        client = Anthropic()
        
        # Define the function for greeting detection
        greeting_function = {
            "name": "detect_greeting",
            "description": "Detect if the input text is a greeting or small talk and provide a friendly response",
            "input_schema": {
                "type": "object",
                "properties": {
                    "is_greeting": {
                        "type": "boolean",
                        "description": "Whether the input is a greeting or small talk"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0 and 1"
                    },
                    "response": {
                        "type": "string",
                        "description": "A friendly response to the greeting"
                    }
                },
                "required": ["is_greeting", "confidence", "response"]
            }
        }

        # Call Anthropic with function calling
        response_message = client.messages.create( # Renamed response to response_message to avoid conflict
            model="claude-3-sonnet-20240229", # Using Sonnet as it's faster and cheaper for this task
            max_tokens=1024, # Can be reduced for this specific task
            messages=[{
                "role": "user",
                "content": f"Analyze if this is a greeting or small talk and provide a friendly response: {query}"
            }],
            tools=[greeting_function]
        )

        # Extract the function call result
        tool_calls = [content for content in response_message.content if content.type == "tool_use"]
        if tool_calls:
            result = tool_calls[0].input
            is_greeting_flag = result.get("is_greeting", False) # Renamed to avoid conflict
            confidence = result.get("confidence", 0.0)
            greeting_response_text = result.get("response", "") # Renamed
            
            # Only consider it a greeting if confidence is high enough
            # The original threshold was 0.7, keeping it unless specified otherwise
            logging.info(f"Greeting detection for query '{query[:50]}...': is_greeting={is_greeting_flag}, confidence={confidence}, response='{greeting_response_text}'")
            return (is_greeting_flag and confidence > 0.7, greeting_response_text)
            
        logging.info(f"No tool_calls for greeting detection from query: {query[:50]}...")
        return (False, "")
    except Exception as e:
        logging.error(f"Error in greeting detection: {e}", exc_info=True)
        return (False, "")

def get_greeting_response() -> str:
    """Generate a friendly greeting response."""
    greetings = [
        "Hey there! How can I help you with your studies today?",
        "Hi! Ready to tackle some learning together?",
        "Hello! What would you like to learn about?",
        "Hey! I'm here to help you understand your textbook better. What's on your mind?",
        "Hi there! Let's make learning fun. What would you like to know?"
    ]
    return random.choice(greetings)
    
logging.basicConfig(level=logging.INFO)
