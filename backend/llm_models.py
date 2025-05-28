from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Iterator
import os

from .enums import LLMModelType # Relative import

class StreamingLLM(ABC):
    """Abstract base class for streaming LLM models"""
    
    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        """Generate text from a prompt and optional context"""
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model"""
        pass

class OpenAIGPT(StreamingLLM):
    """OpenAI GPT model implementation with streaming support"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the OpenAI GPT model"""
        from langchain_openai import ChatOpenAI
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Define system prompt for Pamidi
        jeff_system_prompt = """You are Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
        Your primary goal is to provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
        You should assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
        Focus on delivering precise information based on the context provided.
        Maintain a professional, formal, and knowledgeable tone.
        Always respond as Pamidi - expert, reliable, and focused on assisting finance and tax professionals."""
        
        self._model = ChatOpenAI(model_name=model_name, streaming=False)
        self._streaming_model = ChatOpenAI(model_name=model_name, streaming=True)
        self._jeff_system_prompt = jeff_system_prompt
        self._model_name = model_name
    
    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        """Generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                return self._model.invoke(prompt).content
        else:
            # Set system prompt for non-evaluation mode
            self._model.system = self._jeff_system_prompt
            if context:
                template = """
                Answer the question as Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
                Provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
                Assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
                Focus on delivering precise information based on the context provided.
                Maintain a professional, formal, and knowledgeable tone.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                return self._model.invoke(prompt).content
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._model
        response = chain.invoke({"context": context, "question": prompt})
        return response.content
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        else:
            # Set system prompt for non-evaluation mode
            self._streaming_model.system = self._jeff_system_prompt
            if context:
                template = """
                Answer the question as Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
                Provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
                Assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
                Focus on delivering precise information based on the context provided.
                Maintain a professional, formal, and knowledgeable tone.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._streaming_model
        
        for chunk in chain.stream({"context": context, "question": prompt}):
            yield chunk.content

class GeminiLLM(StreamingLLM):
    """Google Gemini model implementation with streaming support"""
    
    def __init__(self):
        """Initialize the Google Gemini model"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("Gemini API key not found in environment variables")
        
        # Define system prompt for Pamidi
        jeff_system_prompt = """You are Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
        Your primary goal is to provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
        You should assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
        Focus on delivering precise information based on the context provided.
        Maintain a professional, formal, and knowledgeable tone.
        Always respond as Pamidi - expert, reliable, and focused on assisting finance and tax professionals."""
        
        self._model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-preview-05-06", 
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            streaming=False
        )
        
        self._streaming_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-preview-05-06", 
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            streaming=True
        )
        
        self._jeff_system_prompt = jeff_system_prompt
        self._model_name = "gemini-2.5-pro-preview-05-06"
    
    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        """Generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                return self._model.invoke(prompt).content
        else:
            # Set system prompt for non-evaluation mode
            self._model.system_instruction = self._jeff_system_prompt
            if context:
                template = """
                Answer the question as Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
                Provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
                Assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
                Focus on delivering precise information based on the context provided.
                Maintain a professional, formal, and knowledgeable tone.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                return self._model.invoke(prompt).content
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._model
        response = chain.invoke({"context": context, "question": prompt})
        return response.content
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        else:
            # Set system prompt for non-evaluation mode
            self._streaming_model.system_instruction = self._jeff_system_prompt
            if context:
                template = """
                Answer the question as Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
                Provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
                Assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
                Focus on delivering precise information based on the context provided.
                Maintain a professional, formal, and knowledgeable tone.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._streaming_model
        
        for chunk in chain.stream({"context": context, "question": prompt}):
            yield chunk.content

class ClaudeLLM(StreamingLLM):
    """Anthropic Claude model implementation with streaming support"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20240229"):
        """Initialize the Anthropic Claude model"""
        from langchain_anthropic import ChatAnthropic
        
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not found in environment variables")
        
        jeff_system_prompt = """You are Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
        Your primary goal is to provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
        You should assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
        Focus on delivering precise information based on the context provided.
        Maintain a professional, formal, and knowledgeable tone.
        Always respond as Pamidi - expert, reliable, and focused on assisting finance and tax professionals."""
        
        self._model = ChatAnthropic(model=model_name, system=jeff_system_prompt, streaming=False)
        self._streaming_model = ChatAnthropic(model=model_name, system=jeff_system_prompt, streaming=True)
        self._jeff_system_prompt = jeff_system_prompt
        self._model_name = model_name
    
    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        """Generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            # Create a new model instance without the system prompt
            from langchain_anthropic import ChatAnthropic
            evaluation_model = ChatAnthropic(model=self._model.model, streaming=False)
        
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
                prompt_template = ChatPromptTemplate.from_template(template)
                chain = prompt_template | evaluation_model
                response = chain.invoke({"context": context, "question": prompt})
                return response.content
            else:
                return evaluation_model.invoke(prompt).content
        else:
            if context:
                template = """
                Answer the question as Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
                Provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
                Assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
                Focus on delivering precise information based on the context provided.
                Maintain a professional, formal, and knowledgeable tone.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
                prompt_template = ChatPromptTemplate.from_template(template)
                chain = prompt_template | self._model
                response = chain.invoke({"context": context, "question": prompt})
                return response.content
            else:
                return self._model.invoke(prompt).content
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            # Create a new model instance without the system prompt
            from langchain_anthropic import ChatAnthropic
            evaluation_streaming_model = ChatAnthropic(model=self._model.model, streaming=True)
        
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
                prompt_template = ChatPromptTemplate.from_template(template)
                chain = prompt_template | evaluation_streaming_model
                
                for chunk in chain.stream({"context": context, "question": prompt}):
                    yield chunk.content
            else:
                for chunk in evaluation_streaming_model.stream(prompt):
                    yield chunk.content
        else:
            if context:
                template = """
                Answer the question as Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
                Provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
                Assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
                Focus on delivering precise information based on the context provided.
                Maintain a professional, formal, and knowledgeable tone.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
                prompt_template = ChatPromptTemplate.from_template(template)
                chain = prompt_template | self._streaming_model
                
                for chunk in chain.stream({"context": context, "question": prompt}):
                    yield chunk.content
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content

class MistralLLM(StreamingLLM):
    """Mistral model implementation with streaming support"""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        """Initialize the Mistral model"""
        from langchain_mistralai import ChatMistralAI
        
        if not os.environ.get("MISTRAL_API_KEY"):
            raise ValueError("Mistral API key not found in environment variables")
        
        jeff_system_prompt = """You are Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
        Your primary goal is to provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
        You should assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
        Focus on delivering precise information based on the context provided.
        Maintain a professional, formal, and knowledgeable tone.
        Always respond as Pamidi - expert, reliable, and focused on assisting finance and tax professionals."""
        
        self._model = ChatMistralAI(model=model_name, streaming=False)
        self._streaming_model = ChatMistralAI(model=model_name, streaming=True)
        self._jeff_system_prompt = jeff_system_prompt
        self._model_name = model_name
    
    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        """Generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                return self._model.invoke(prompt).content
        else:
            # Set system prompt for non-evaluation mode
            self._model.system = self._jeff_system_prompt
            if context:
                template = """
                Answer the question as Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
                Provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
                Assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
                Focus on delivering precise information based on the context provided.
                Maintain a professional, formal, and knowledgeable tone.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                return self._model.invoke(prompt).content
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._model
        response = chain.invoke({"context": context, "question": prompt})
        return response.content
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        else:
            # Set system prompt for non-evaluation mode
            self._streaming_model.system = self._jeff_system_prompt
            if context:
                template = """
                Answer the question as Pamidi, a specialized AI assistant for Chartered Accountants and tax experts.
                Provide concise, accurate, and up-to-date information regarding tax laws, accounting standards, and financial regulations.
                Assist with queries related to tax planning, compliance, audit procedures, and interpretation of financial statements.
                Focus on delivering precise information based on the context provided.
                Maintain a professional, formal, and knowledgeable tone.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._streaming_model
        
        for chunk in chain.stream({"context": context, "question": prompt}):
            yield chunk.content

class LLMFactory:
    """Factory for creating LLM models (Factory Pattern)"""
    
    @staticmethod
    def create_llm(model_type: LLMModelType) -> StreamingLLM:
        """Create an LLM model based on the model type"""
        if model_type == LLMModelType.OPENAI_GPT35:
            return OpenAIGPT(model_name="gpt-3.5-turbo")
        elif model_type == LLMModelType.OPENAI_GPT4:
            return OpenAIGPT(model_name="gpt-4")
        elif model_type == LLMModelType.GEMINI:
            return GeminiLLM()
        elif model_type == LLMModelType.CLAUDE_3_OPUS:
            return ClaudeLLM(model_name="claude-3-opus-20240229") 
        elif model_type == LLMModelType.CLAUDE_37_SONNET:
            return ClaudeLLM(model_name="claude-3-7-sonnet-20250219")
        elif model_type == LLMModelType.MISTRAL_LARGE:
            return MistralLLM(model_name="mistral-large-latest")
        elif model_type == LLMModelType.MISTRAL_MEDIUM:
            return MistralLLM(model_name="mistral-medium-latest")
        elif model_type == LLMModelType.MISTRAL_SMALL:
            return MistralLLM(model_name="mistral-small-latest")
        else:
            raise ValueError(f"Unsupported LLM model: {model_type}")