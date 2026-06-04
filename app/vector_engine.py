import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# --- Abstract Base Class to enforce structure ---

class EmbeddingProvider(ABC):
    """Abstract interface for different embedding providers."""
    @abstractmethod
    def generate(self, texts: List[str]) -> np.ndarray:
        pass

# --- Concrete Implementations ---

class LocalEmbeddingProvider(EmbeddingProvider):
    """Uses Sentence-Transformers for local, CPU/GPU processing."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def generate(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

class CloudEmbeddingProvider(EmbeddingProvider):
    """Uses OpenAI API for cloud-based embeddings."""
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, texts: List[str]) -> np.ndarray:
        # OpenAI expects a list of strings
        response = self.client.embeddings.create(input=texts, model=self.model)
        # Extract embeddings and convert to numpy array
        return np.array([data.embedding for data in response.data])

# --- Main Engine Class ---

class VectorEngine:
    """
    The main engine that handles chunking, embedding orchestration, 
    and similarity scoring.
    """
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Splits text into overlapping chunks to maintain context.
        """
        words = text.split()
        chunks = []
        
        # If text is too small, just return one chunk
        if len(words) <= chunk_size:
            return [text]

        i = 0
        while i < len(words):
            # Take a slice of words
            chunk_slice = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_slice))
            
            # Move index forward, but subtract overlap to keep context
            i += (chunk_size - overlap)
            
            # Break if we've reached the end of the list
            if i >= len(words):
                break
        return chunks

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Delegates embedding generation to the chosen provider."""
        return self.provider.generate(texts)

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculates Cosine Similarity between two vectors."""
        # Normalize vectors to unit length for cosine similarity via dot product
        norm_a = vec_a / np.linalg.norm(vec_a)
        norm_b = vec_b / np.linalg.norm(vec_b)
        return float(np.dot(norm_a, norm_b))

    def rank_similarities(self, query_embedding: np.ndarray, document_embeddings: np.ndarray, metadata: List[Dict]) -> List[Dict]:
        """
        Scores and ranks a list of documents against a single query embedding.
        Returns a list of dicts containing scores and original metadata.
        """
        results = []
        for i, doc_vec in enumerate(document_embeddings):
            score = self.compute_similarity(query_embedding, doc_vec)
            # Merge score with metadata for easy identification
            result_item = {**metadata[i], "similarity_score": score}
            results.append(result_item)

        # Sort by highest score first
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
