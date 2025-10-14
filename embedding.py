import logger
import requests
from langchain.schema.embeddings import Embeddings


class EmbeddingClass(Embeddings):
    """
    Custom embeddings class using EmbeddingGemma-300M via Ollama.
    
    This class implements task-specific prompts for optimal retrieval performance:
    - Query prompts: "task: search result | query: {text}"
    - Document prompts: "title: none | text: {text}"
    """
    
    def __init__(self, model_name: str = "embeddinggemma:300m", base_url: str = "http://localhost:11434"):
        """
        Initialize EmbeddingGemma embeddings via Ollama.
        
        Args:
            model_name (str): Ollama model name for EmbeddingGemma
            base_url (str): Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embed"
        
        logger.info(f"Initializing EmbeddingGemma via Ollama: {model_name}")
        logger.info(f"Ollama API: {self.api_url}")
        
        try:
            # Test connection to Ollama
            self._test_ollama_connection()
            logger.info("✅ EmbeddingGemma model ready via Ollama")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {str(e)}")
            self._show_ollama_help()
            raise
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available."""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": "test"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return True
            elif response.status_code == 404:
                raise Exception(f"Model {self.model_name} not found. Please run: ollama pull {self.model_name}")
            else:
                raise Exception(f"Ollama returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama - is it running?")
        except requests.exceptions.Timeout:
            raise Exception("Ollama connection timeout")
            
    def _show_ollama_help(self):
        """Show Ollama setup help message."""
        logger.error("🔑 Ollama Setup Required!")
        logger.error("To use EmbeddingGemma with Ollama, you need to:")
        logger.error("1. Make sure Ollama is running: ollama serve")
        logger.error(f"2. Pull the model: ollama pull {self.model_name}")
        logger.error(f"3. Verify Ollama is accessible at: {self.base_url}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using the document-specific prompt format.
        
        Args:
            texts (List[str]): List of document texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            # Format with EmbeddingGemma document prompt template
            formatted_text = f"title: none | text: {text}"
            
            # Generate embedding via Ollama API
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": formatted_text
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    embedding = response.json()["embedding"]
                    embeddings.append(embedding)
                else:
                    logger.error(f"Failed to generate embedding: {response.status_code}")
                    # Return zero vector as fallback
                    embeddings.append([0.0] * 768)
                    
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query using the query-specific prompt format.
        
        Args:
            text (str): Query text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        # Format text with query prompt
        formatted_text = f"task: search result | query: {text}"
        
        # Generate embedding via Ollama API
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": formatted_text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                logger.error(f"Failed to generate query embedding: {response.status_code}")
                return [0.0] * 768
                
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return [0.0] * 768

