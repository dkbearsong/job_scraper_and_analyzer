import os
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# =============================================
# KEYWORD ADJUSTMENTS CONFIGURATION
# =============================================
def _load_env_json(key: str, default: str) -> dict:
    """Load a JSON dict from an env var, falling back to the provided default."""
    raw = os.getenv(key, "")
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return json.loads(default)

KEYWORD_ADJUSTMENTS = _load_env_json(
    "KEYWORD_ADJUSTMENTS",
    '{"preferred": {"python": 0.03, "rust": 0.02, "kubernetes": 0.025}, "penalty": {"wordpress": -0.08, "wix": -0.05, "legacy": -0.03}}'
)

# =============================================
# METADATA ADJUSTMENTS CONFIGURATION
# =============================================
METADATA_ADJUSTMENTS = _load_env_json(
    "METADATA_ADJUSTMENTS",
    '{"remote_bonus": 0.02, "salary_threshold": 90000, "salary_bonus": 0.03, "recency_days": 30, "recency_bonus": 0.02}'
)


def apply_keyword_adjustments(base_score: float, skills_list: List[str], title: str) -> float:
    """
    Apply keyword-based bonuses and penalties to a semantic score.
    Case-insensitive matching against skills list and job title.
    """
    score = base_score
    skills_lower = [s.lower() for s in skills_list]
    title_lower = title.lower()
    
    # Apply preferred bonuses
    for kw, delta in KEYWORD_ADJUSTMENTS["preferred"].items():
        if kw in skills_lower or kw in title_lower:
            score += delta
            
    # Apply penalty deductions
    for kw, delta in KEYWORD_ADJUSTMENTS["penalty"].items():
        if kw in skills_lower or kw in title_lower:
            score += delta
            
    return score


def apply_metadata_adjustments(base_score: float, job_meta: Dict[str, Any]) -> float:
    """
    Apply metadata-based bonuses to a semantic score.
    Checks for remote work, salary threshold, and job recency.
    """
    score = base_score
    if job_meta.get("is_remote", False):
        score += METADATA_ADJUSTMENTS["remote_bonus"]
    if job_meta.get("salary", 0) >= METADATA_ADJUSTMENTS["salary_threshold"]:
        score += METADATA_ADJUSTMENTS["salary_bonus"]
    if job_meta.get("days_old", 999) <= METADATA_ADJUSTMENTS["recency_days"]:
        score += METADATA_ADJUSTMENTS["recency_bonus"]
    return score

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
