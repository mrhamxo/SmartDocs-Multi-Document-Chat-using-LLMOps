import os
import json
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from multi_doc_chat.logger.logger import CustomLogger as log
from multi_doc_chat.exception.exception import DocumentPortalException
from utils.config_loader import load_config
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import os
import json
from dotenv import load_dotenv

class ApiKeyManager:
    """
    Manages API keys required for the application.
    
    Features:
        - Loads API keys from a single environment variable (JSON string) if available.
        - Falls back to individual environment variables for missing keys.
        - Validates that all required keys are present.
        - Provides a method to retrieve individual keys safely.
    """

    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY"]

    def __init__(self):
        """
        Initialize the API key manager by loading keys from environment variables.
        
        Raises:
            DocumentPortalException: If any required API keys are missing.
        """
        self.api_keys = {}
        # Attempt to load all API keys from a single JSON env variable
        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Fallback: load missing keys individually from environment
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # Validate all required keys are present
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

        # Log loaded keys (partial display for security)
        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})

    def get(self, key: str) -> str:
        """
        Retrieve a specific API key by name.
        
        Args:
            key (str): Name of the API key to retrieve.
        
        Returns:
            str: The API key value.
        
        Raises:
            KeyError: If the requested key is missing.
        """
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    """
    Loads embedding and LLM models using configuration and API keys.
    
    Features:
        - Loads YAML configuration file using `load_config()`.
        - Automatically loads .env in local development.
        - Provides methods to load embedding models and LLMs from supported providers.
    """

    def __init__(self):
        """
        Initialize the ModelLoader.
        
        Steps:
            - Load .env file in local mode.
            - Instantiate ApiKeyManager to manage API keys.
            - Load YAML configuration.
        """
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """
        Load and return the configured embedding model.

        Returns:
            GoogleGenerativeAIEmbeddings: Embedding model instance.
        
        Raises:
            DocumentPortalException: If loading the embedding model fails.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY")  # type: ignore
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model based on the provider.
        
        Returns:
            LLM instance (ChatGoogleGenerativeAI, ChatGroq, etc.)
        
        Raises:
            ValueError: If the requested LLM provider is missing or unsupported.
        """
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "google")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("   "),
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"),  # type: ignore
                temperature=temperature,
            )
            
        # elif provider == "openai":
        #     return ChatOpenAI(
        #         model=model_name,
        #         api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    # Initialize ModelLoader for testing
    loader = ModelLoader()

    # Test embedding model
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM model
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.Content}")