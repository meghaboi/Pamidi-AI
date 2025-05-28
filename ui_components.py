import streamlit as st
import logging
import time
from utils import text_to_speech, is_greeting, check_api_keys, save_uploaded_file
from enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from pipeline_utils import initialize_pipeline

def display_chat_interface():
    st.header("💬 Chat with Pamidi")
    st.markdown("Hey! Got questions about your textbook? Lay 'em on me. I'll break it down for ya.")

    # File uploader
    uploaded_file = st.file_uploader("Upload your textbook (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # Check if this is a new file upload
        if uploaded_file.name != st.session_state.get('last_uploaded_filename'):
            # Save the uploaded file and get its path
            temp_file_path = save_uploaded_file(uploaded_file)
            if temp_file_path:
                st.session_state.file_path = temp_file_path # Store the path
                st.session_state.last_uploaded_filename = uploaded_file.name
                st.session_state.messages = [] # Clear messages on new file
                st.session_state.pipeline = None # Reset pipeline

                # Initialize pipeline as soon as file is uploaded
                with st.spinner("🧠 Pamidi is ingesting your textbook... This might take a moment."):
                    try:
                        st.session_state.pipeline = initialize_pipeline(
                            file_path=st.session_state.file_path, # Use the saved file path
                            embedding_model_enum=EmbeddingModelType.from_string(st.session_state.embedding_model),
                            vector_store_enum=VectorStoreType.from_string(st.session_state.vector_store),
                            reranker_enum=RerankerModelType.from_string(st.session_state.reranker),
                            llm_enum=LLMModelType.from_string(st.session_state.llm_model),
                            chunking_strategy_enum=ChunkingStrategyType.from_string(st.session_state.chunking_strategy),
                            hybrid_alpha=st.session_state.hybrid_alpha,
                            chunk_size=st.session_state.chunk_size,
                            chunk_overlap=st.session_state.chunk_overlap,
                            top_k=st.session_state.top_k
                        )
                        st.success("Textbook loaded! I'm ready for your questions.")
                        # Add a welcome message after initialization
                        welcome_msg = "Alright, let's get this study session started! What's on your mind?"
                        welcome_audio_bytes = text_to_speech(welcome_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": welcome_msg,
                            "audio": welcome_audio_bytes,
                            "contexts": [],
                            "elapsed_time": None
                        })
                    except Exception as e:
                        st.error(f"Error initializing Pamidi: {e}")
                        logging.error(f"Pipeline initialization failed: {e}", exc_info=True)
                        st.session_state.pipeline = None # Ensure pipeline is None if init fails
            else:
                st.error("Failed to save uploaded file. Please try again.")
                st.session_state.pipeline = None
                st.session_state.file_path = None
                st.session_state.last_uploaded_filename = None
        elif st.session_state.pipeline is None and st.session_state.file_path:
            # This case handles if initialization failed previously and user hasn't uploaded a NEW file.
            # We can try to re-initialize or just inform them. For now, inform.
            st.warning("Looks like the textbook didn't load correctly. Try re-uploading the file or a different one.")


    if not st.session_state.messages and st.session_state.pipeline is None and not uploaded_file: # Adjusted initial message condition
         # This message shows only if no file is uploaded yet.
         initial_message = "Upload your textbook to get started!"
         st.info(initial_message)
         # Optionally, you could add this to st.session_state.messages if you want it in chat history
         # For now, it's just an st.info message.

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                response_text = message.get("content")
                audio_data = message.get("audio") 
                contexts = message.get("contexts", [])
                elapsed_time = message.get("elapsed_time")

                if response_text:
                    tab_labels = ["📖 Read Response", "🔊 Hear Response"]
                    try:
                        tab_text, tab_audio = st.tabs(tab_labels)
                    except Exception as e:
                        logging.error(f"Error creating tabs: {e}")
                        st.write(response_text) 
                        if audio_data: st.audio(audio_data, format="audio/mp3")
                        tab_text = None 

                    if tab_text: 
                        with tab_text:
                            st.write(response_text)

                        with tab_audio:
                            if audio_data:
                                st.audio(audio_data, format="audio/mp3")
                            else:
                                st.info("Audio playback is not available for this message.")

                    if elapsed_time is not None:
                         st.write(f"_(Pamidi cooked that up in {elapsed_time:.2f} seconds)_")

                    if st.session_state.show_contexts and contexts:
                         with st.expander("🧠 Check out the textbook bits I used:"):
                            for i, context in enumerate(contexts):
                                st.markdown(f"**Snippet {i+1}:**")
                                st.text(context)

                else: 
                    st.write("*Assistant message content missing.*")

            else: 
                st.write(message["content"])

    user_query = st.chat_input("Type your question here...")

    if user_query:
        logging.info(f"User query received: {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        if st.session_state.pipeline is None:
            logging.warning("Chat query received, but pipeline not initialized.")
            warning_msg = "Whoa there! Looks like we haven't loaded your textbook into my brain yet. Please upload it first!"
            warning_audio = text_to_speech(warning_msg)
            with st.chat_message("assistant"):
                 tab_labels_warn = ["📖 Read Message", "🔊 Hear Message"]
                 tab_warn_text, tab_warn_audio = st.tabs(tab_labels_warn)
                 with tab_warn_text: st.warning(warning_msg, icon="✋")
                 with tab_warn_audio:
                     if warning_audio: st.audio(warning_audio, format="audio/mp3")
                     else: st.info("Audio playback not available.")

            st.session_state.messages.append({
                "role": "assistant", "content": warning_msg, "audio": warning_audio,
                "contexts": [], "elapsed_time": None
            })
            st.stop()

        # Check if it's a greeting
        is_greet, greeting_response = is_greeting(user_query)
        if is_greet:
            greeting_audio = text_to_speech(greeting_response)
            
            with st.chat_message("assistant"):
                tab_labels_greet = ["📖 Read Response", "🔊 Hear Response"]
                tab_greet_text, tab_greet_audio = st.tabs(tab_labels_greet)
                
                with tab_greet_text:
                    st.write(greeting_response)
                
                with tab_greet_audio:
                    if greeting_audio:
                        st.audio(greeting_audio, format="audio/mp3")
                    else:
                        st.info("Audio playback not available.")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": greeting_response,
                    "audio": greeting_audio,
                    "contexts": [],
                    "elapsed_time": None
                })
            return

        # Start streaming response process for non-greeting queries
        with st.chat_message("assistant"):
            start_time = time.time()
            
            try:
                logging.info("Fetching contexts from vector store...")
                contexts = st.session_state.pipeline.retrieve_context(user_query)
                
                tab_labels_stream = ["📖 Read Response", "🔊 Hear Response"]
                tab_stream_text, tab_stream_audio = st.tabs(tab_labels_stream)
                
                with tab_stream_text:
                    stream_placeholder = st.empty()
                
                with tab_stream_audio:
                    audio_placeholder = st.empty()
                    audio_placeholder.info("Audio will be available when response is complete.")
                
                logging.info("Starting streaming generation...")
                full_response = ""
                
                for chunk in st.session_state.pipeline.stream_run(user_query):
                    if chunk is not None:
                        full_response += chunk
                        with tab_stream_text:
                            stream_placeholder.markdown(full_response + "▌")
                    else:
                        logging.warning("Received None chunk from stream_run, skipping")
                
                with tab_stream_text:
                    stream_placeholder.markdown(full_response)
                
                elapsed_time = time.time() - start_time
                st.write(f"_(Pamidi cooked that up in {elapsed_time:.2f} seconds)_")
                
                logging.info("Generating TTS audio for the complete response...")
                tts_start_time = time.time()
                audio_bytes = text_to_speech(full_response)
                tts_elapsed_time = time.time() - tts_start_time
                
                log_msg = f"TTS generation {'succeeded' if audio_bytes else 'failed/skipped'} in {tts_elapsed_time:.2f}s."
                if audio_bytes: 
                    logging.info(log_msg)
                    with tab_stream_audio:
                        audio_placeholder.audio(audio_bytes, format="audio/mp3")
                else: 
                    logging.warning(log_msg)
                    with tab_stream_audio:
                        audio_placeholder.info("Audio playback is not available for this message.")
                
                if st.session_state.show_contexts and contexts:
                    with st.expander("🧠 Check out the textbook bits I used:"):
                        for i, context in enumerate(contexts):
                            st.markdown(f"**Snippet {i+1}:**")
                            st.text(context)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "contexts": contexts,
                    "elapsed_time": elapsed_time, 
                    "audio": audio_bytes
                })
                
            except Exception as e:
                logging.error(f"Error processing query or generating audio: {e}", exc_info=True)
                error_msg = f"Oof, hit a snag trying to answer that. Maybe try rephrasing? Error: {str(e)}"
                error_audio = text_to_speech(error_msg)
                
                tab_labels_err = ["📖 Read Error", "🔊 Hear Error"]
                tab_err_text, tab_err_audio = st.tabs(tab_labels_err)
                
                with tab_err_text: 
                    st.error(error_msg, icon="🔥")
                
                with tab_err_audio:
                    if error_audio: 
                        st.audio(error_audio, format="audio/mp3")
                    else: 
                        st.info("Audio playback not available.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg, 
                    "audio": error_audio,
                    "contexts": [], 
                    "elapsed_time": None
                })

# Removed display_evaluation_interface function and all its content
# Ensure no trailing code from the old display_evaluation_interface function exists 