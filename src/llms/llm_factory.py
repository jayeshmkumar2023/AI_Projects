import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from src.config import settings

class LLMFactory:
    @staticmethod
    def get_llm(provider: str = "groq", model_name: str = None):
        """
        Returns the configured LangChain Chat LLM.
        Supported providers: 'groq', 'openai'.
        """
        provider = provider.lower()
        if provider == "groq":
            api_key = settings.GROQ_API_KEY
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
            
            # Default to Llama 3.3 70b or 3.1 8b
            model = model_name or "llama-3.3-70b-versatile"
            return ChatGroq(
                api_key=api_key,
                model=model,
                temperature=0.2
            )
            
        elif provider == "openai":
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment or .env file.")
                
            model = model_name or "gpt-4o-mini"
            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.2
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Choose 'groq' or 'openai'.")
