document.addEventListener('DOMContentLoaded', () => {
    // 1. DOM Element References
    const fileUploadInput = document.getElementById('fileUpload');
    const uploadButton = document.getElementById('uploadButton');
    const uploadStatusDiv = document.getElementById('uploadStatus');
    const chatbox = document.getElementById('chatbox');
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    const showContextsToggle = document.getElementById('showContextsToggle');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // 2. Global State Variable
    let currentSessionId = null;

    // Initial State
    chatInput.disabled = true;
    sendButton.disabled = true;

    // 3. Event Listener for File Upload
    uploadButton.addEventListener('click', async () => {
        const file = fileUploadInput.files[0];
        if (!file) {
            uploadStatusDiv.textContent = 'Please select a file first.';
            uploadStatusDiv.className = 'status-error';
            return;
        }

        loadingIndicator.style.display = 'block';
        uploadStatusDiv.textContent = 'Processing file...';
        uploadStatusDiv.className = '';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (response.ok) {
                currentSessionId = result.session_id;
                uploadStatusDiv.textContent = `File '${result.filename}' processed. Session ID: ${currentSessionId}. Ready to chat!`;
                uploadStatusDiv.className = 'status-success';
                chatbox.innerHTML = ''; // Clear chatbox
                chatInput.disabled = false;
                sendButton.disabled = false;
            } else {
                uploadStatusDiv.textContent = `Error: ${result.detail || 'File processing failed.'}`;
                uploadStatusDiv.className = 'status-error';
                currentSessionId = null;
                chatInput.disabled = true;
                sendButton.disabled = true;
            }
        } catch (error) {
            console.error('Upload error:', error);
            uploadStatusDiv.textContent = 'Error: Could not connect to the server or an unexpected error occurred.';
            uploadStatusDiv.className = 'status-error';
            currentSessionId = null;
            chatInput.disabled = true;
            sendButton.disabled = true;
        } finally {
            loadingIndicator.style.display = 'none';
        }
    });
    
    // 4. Event Listener for Send Button and Enter Key
    sendButton.addEventListener('click', () => {
        const query = chatInput.value.trim();
        if (query) {
            processQuery(query);
        }
    });

    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            const query = chatInput.value.trim();
            if (query) {
                processQuery(query);
            }
        }
    });

    function processQuery(query) {
        if (!currentSessionId) {
            addMessageToChatbox("Please upload and process a file first.", "system");
            return;
        }
        chatInput.value = '';
        addMessageToChatbox(query, 'user');
        loadingIndicator.style.display = 'block';
        handleUserQuery(query);
    }

    // 5. handleUserQuery(query) Function
    async function handleUserQuery(query) {
        try {
            // Check for Greeting
            const greetingResponse = await fetch('/api/check_greeting', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query }),
            });
            
            if (greetingResponse.ok) {
                const greetingResult = await greetingResponse.json();
                if (greetingResult.is_greeting) {
                    addMessageToChatbox(greetingResult.response, 'assistant', 0, [], true);
                    loadingIndicator.style.display = 'none';
                    return;
                }
            } else {
                console.warn('Greeting check failed or API not available, proceeding with chat.');
            }

            // If not a greeting, proceed to chat
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, query: query }),
            });

            if (!response.ok) {
                const errorResult = await response.json();
                addMessageToChatbox(`Error: ${errorResult.detail || 'Chat API error'}`, 'system');
                loadingIndicator.style.display = 'none';
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            let assistantMessageElement = addMessageToChatbox("", "assistant", 0, [], false);
            const messageTextContentElement = assistantMessageElement.querySelector('.message-text-content');
            let fullResponse = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const sseMessages = decoder.decode(value, { stream: true }).split('\n\n').filter(Boolean);
                
                for (const sseMessage of sseMessages) {
                    if (sseMessage.startsWith('data: ')) {
                        try {
                            const jsonData = JSON.parse(sseMessage.substring(6));
                            if (jsonData.type === 'chunk') {
                                fullResponse += jsonData.data;
                                messageTextContentElement.textContent = fullResponse + '▌'; // Blinking cursor
                            } else if (jsonData.type === 'result') {
                                fullResponse = jsonData.full_response;
                                messageTextContentElement.textContent = fullResponse;
                                
                                const timeMeta = assistantMessageElement.querySelector('.message-meta span:last-child');
                                if (timeMeta) timeMeta.textContent = `Time: ${jsonData.elapsed_time.toFixed(2)}s`;
                                
                                displayContexts(assistantMessageElement, jsonData.contexts);
                                
                                const playBtn = assistantMessageElement.querySelector('.play-audio-btn');
                                if (playBtn) {
                                    playBtn.disabled = false;
                                    playBtn.onclick = () => playMessageAudio(assistantMessageElement, fullResponse);
                                }
                                reader.cancel(); // Close the stream
                                loadingIndicator.style.display = 'none';
                                return; 
                            } else if (jsonData.type === 'error') {
                                console.error('SSE Error:', jsonData.data);
                                addMessageToChatbox(`Error from server: ${jsonData.data}`, 'system');
                                messageTextContentElement.textContent = fullResponse + `\n[Error: ${jsonData.data}]`;
                                reader.cancel();
                                loadingIndicator.style.display = 'none';
                                return;
                            }
                        } catch (e) {
                            console.error("Error parsing SSE JSON:", e, "Message:", sseMessage);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Chat handling error:', error);
            addMessageToChatbox(`Error: ${error.message || 'Failed to get response.'}`, 'system');
        } finally {
            loadingIndicator.style.display = 'none';
             // Ensure cursor is removed if loop finishes unexpectedly
            const tempAssistantMsg = chatbox.querySelector('.assistant-message .message-text-content');
            if (tempAssistantMsg && tempAssistantMsg.textContent.endsWith('▌')) {
                tempAssistantMsg.textContent = tempAssistantMsg.textContent.slice(0, -1);
            }
        }
    }

    // 6. addMessageToChatbox Function
    function addMessageToChatbox(text, sender, time = 0, contexts = [], playAudioImmediately = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        
        const textContentP = document.createElement('p');
        textContentP.classList.add('message-text-content'); // Added class for easier selection
        textContentP.textContent = text;
        contentDiv.appendChild(textContentP);

        messageDiv.appendChild(contentDiv);

        const metaDiv = document.createElement('div');
        metaDiv.classList.add('message-meta');
        const senderSpan = document.createElement('span');
        senderSpan.textContent = sender.charAt(0).toUpperCase() + sender.slice(1);
        metaDiv.appendChild(senderSpan);

        if (sender === 'assistant') {
            const timeSpan = document.createElement('span');
            timeSpan.textContent = ` | Time: ${time.toFixed(2)}s`;
            metaDiv.appendChild(timeSpan);
            
            const playButton = document.createElement('button');
            playButton.classList.add('play-audio-btn');
            playButton.textContent = 'Hear Response';
            playButton.disabled = !text || text.trim() === ""; // Disable if no text initially
            contentDiv.appendChild(playButton);
            
            const audioPlayer = document.createElement('audio');
            audioPlayer.classList.add('message-audio');
            audioPlayer.style.display = 'none'; // Hidden by default
            contentDiv.appendChild(audioPlayer);

            playButton.onclick = () => playMessageAudio(messageDiv, textContentP.textContent); // Use current text content

            const contextsDiv = document.createElement('div');
            contextsDiv.classList.add('contexts');
            contextsDiv.style.display = showContextsToggle.checked ? 'block' : 'none';
            const contextsHeader = document.createElement('h4');
            contextsHeader.textContent = 'Contexts Used:';
            contextsDiv.appendChild(contextsHeader);
            messageDiv.appendChild(contextsDiv);

            if (contexts && contexts.length > 0) {
                displayContexts(messageDiv, contexts);
            }
            
            if (playAudioImmediately && text && text.trim() !== "") {
                playMessageAudio(messageDiv, text);
            }
        }
        messageDiv.appendChild(metaDiv);
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
        return messageDiv;
    }

    // 7. playMessageAudio Function
    async function playMessageAudio(messageElement, text) {
        const audioPlayer = messageElement.querySelector('.message-audio');
        const playButton = messageElement.querySelector('.play-audio-btn');
        
        if (!text || text.trim() === "") {
            console.warn("No text to synthesize for audio.");
            return;
        }

        if (audioPlayer.src && audioPlayer.src !== window.location.href) { // Check if src is already set and not just base URL
            audioPlayer.play();
            return;
        }
        
        const originalButtonText = playButton.textContent;
        playButton.textContent = 'Loading...';
        playButton.disabled = true;

        try {
            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text }),
            });

            if (response.ok) {
                const blob = await response.blob();
                audioPlayer.src = URL.createObjectURL(blob);
                audioPlayer.play();
            } else {
                const errorResult = await response.json();
                console.error('TTS API Error:', errorResult.detail);
                addMessageToChatbox(`TTS Error: ${errorResult.detail}`, 'system');
            }
        } catch (error) {
            console.error('TTS fetch error:', error);
            addMessageToChatbox('TTS Error: Could not connect or process audio.', 'system');
        } finally {
            playButton.textContent = originalButtonText;
            playButton.disabled = false;
        }
    }

    // 8. displayContexts Function
    function displayContexts(messageElement, contexts) {
        const contextsDiv = messageElement.querySelector('.contexts');
        if (!contextsDiv) return;

        const contextsHeader = contextsDiv.querySelector('h4');
        contextsDiv.innerHTML = ''; // Clear previous
        contextsDiv.appendChild(contextsHeader); // Re-add header

        if (contexts && contexts.length > 0) {
            contexts.forEach(contextText => {
                const snippetDiv = document.createElement('div');
                snippetDiv.classList.add('context-snippet');
                snippetDiv.textContent = contextText;
                contextsDiv.appendChild(snippetDiv);
            });
        } else {
            const noContextP = document.createElement('p');
            noContextP.textContent = "No specific contexts were highlighted for this response.";
            contextsDiv.appendChild(noContextP);
        }
        contextsDiv.style.display = showContextsToggle.checked ? 'block' : 'none';
    }

    // 9. Event Listener for Context Toggle
    showContextsToggle.addEventListener('change', () => {
        const allContextsDivs = document.querySelectorAll('.assistant-message .contexts');
        allContextsDivs.forEach(div => {
            div.style.display = showContextsToggle.checked ? 'block' : 'none';
        });
    });
});
