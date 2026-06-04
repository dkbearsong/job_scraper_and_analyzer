import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.vector_engine import VectorEngine, LocalEmbeddingProvider

@dataclass
class Archetype:
    """Represents a profile (e.g., Resume, User Profile) used for comparison."""
    name: str
    type: str  # e.g., 'resume', 'user_profile'
    title_embedding: Optional[np.ndarray] = None
    skills_embedding: Optional[np.ndarray] = None
    responsibilities_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = {}

    def get_combined_embedding(self) -> Optional[np.ndarray]:
        """Returns a combined embedding from available archetype vectors."""
        embeddings = [
            emb for emb in (
                self.title_embedding,
                self.skills_embedding,
                self.responsibilities_embedding,
            )
            if emb is not None
        ]
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)

class ArchetypeManager:
    """
    Handles loading archetype profiles, creating semantic documents for jobs,
    and comparing jobs to archetypes using embeddings.
    """

    def __init__(self, embedding_provider: Optional[LocalEmbeddingProvider] = None):
        """
        Initializes the manager with a VectorEngine.
        If no provider is provided, defaults to LocalEmbeddingProvider.
        """
        if embedding_provider is None:
            embedding_provider = LocalEmbeddingProvider()
        
        self.vector_engine = VectorEngine(embedding_provider)
        self.archetypes: List[Archetype] = []

    def load_archetype(self, name: str, archetype_data: Dict[str, str], archetype_type: str = "benchmark", metadata: Optional[Dict] = None) -> Archetype:
        """
        Loads an archetype by generating the three required embeddings defined in Stage 4.
        archetype_data keys: 'title', 'skills', 'responsibilities'
        """
        title_text = archetype_data.get("title", "")
        skills_text = archetype_data.get("skills", "")
        resp_text = archetype_data.get("responsibilities", "")

        # In a production scenario, you would check a cache/database here 
        # before calling get_embeddings to satisfy the "must be cached" constraint.
        
        embeddings = self.vector_engine.get_embeddings([title_text, skills_text, resp_text])
        
        new_archetype = Archetype(
            name=name,
            type=archetype_type,
            title_embedding=embeddings[0],
            skills_embedding=embeddings[1],
            responsibilities_embedding=embeddings[2],
            metadata=metadata or {}
        )
        self.archetypes.append(new_archetype)
        return new_archetype

    def add_archetype(self, archetype: Archetype):
        """Adds a pre-loaded/cached archetype directly to the manager."""
        self.archetypes.append(archetype)

    def create_semantic_document(self, job_data: Dict[str, Any]) -> str:
        """
        Creates a single semantic document string from job data for embedding.
        Combines title, description, and other key fields to capture the role's essence.
        """
        title = job_data.get("title", "")
        description = job_data.get("description", "")
        # Add other relevant fields if they exist in the job_data
        skills = " ".join(job_data.get("skills", [])) if isinstance(job_data.get("skills"), list) else ""
        
        # Construct a cohesive text block for the embedding model
        document = f"Job Title: {title}. Description: {description} Skills: {skills}"
        return document

    def compare_job_to_archetypes(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compares a single job against all loaded archetypes.
        Returns a list of matches with scores and metadata.
        """
        if not self.archetypes:
            return []

        # 1. Create the semantic document for the job
        job_doc = self.create_semantic_document(job_data)
        
        # 2. Generate the job embedding
        job_embedding = self.vector_engine.get_embeddings([job_doc])[0]
        
        matches = []
        for archetype in self.archetypes:
            archetype_embedding = archetype.get_combined_embedding()
            if archetype_embedding is None:
                continue

            score = self.vector_engine.compute_similarity(job_embedding, archetype_embedding)
            
            matches.append({
                "archetype_name": archetype.name,
                "archetype_type": archetype.type,
                "similarity_score": float(score),
                "metadata": archetype.metadata
            })
        
        # Sort matches by highest score first
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)

    def generate_retrieval_metadata(self, job_data: Dict[str, Any], matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates structured retrieval metadata from archetype comparison results.
        """
        if not matches:
            return {
                "match_count": 0,
                "best_match": None,
                "scores": []
            }

        return {
            "match_count": len(matches),
            "best_match": matches[0]["archetype_name"] if matches else None,
            "scores": [
                {"name": m["archetype_name"], "score": m["similarity_score"]} 
                for m in matches
            ]
        }

    def apply_rule_filters(self, job_data: Dict[str, Any], matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies rule-based filters to archetype matches including pay range filtering.
        Filters out matches that don't satisfy the user's target pay expectations.
        """
        # Extract job pay range if available
        job_pay = job_data.get("features", {}).get("pay", "")
        
        # If no pay information in job, keep all matches (can't filter)
        if not job_pay or job_pay == "Not Specified":
            return matches
            
        filtered_matches = []
        
        for match in matches:
            archetype_metadata = match.get("metadata", {})
            
            # Check if this archetype has pay range information
            min_pay = archetype_metadata.get("pay_range_min")
            max_pay = archetype_metadata.get("pay_range_max")
            
            # If archetype doesn't have pay range info, keep it (can't filter)
            if min_pay is None or max_pay is None:
                filtered_matches.append(match)
                continue
                
            # Try to parse job pay range (simplified - in real system could use more robust parsing)
            try:
                # Extract numeric values from job pay string (e.g. "$50,000 - $70,000" -> 50000-70000)
                job_min, job_max = self._parse_job_pay_range(job_pay)
                
                # Apply filtering logic
                if job_min is not None and job_max is not None:
                    # Keep match if the archetype's target pay range overlaps with job pay
                    if min_pay <= job_max and max_pay >= job_min:
                        filtered_matches.append(match)
                else:
                    # If we can't parse the job pay properly, keep match (can't filter)
                    filtered_matches.append(match)
                    
            except Exception:
                # If parsing fails, keep the match (safe default)
                filtered_matches.append(match)
        
        return filtered_matches
    
    def _parse_job_pay_range(self, pay_text: str) -> tuple:
        """
        Parses a job's pay text to extract min and max values.
        This is a simplified implementation - in practice this could use more robust parsing
        to handle various formats like "$50,000 - $70,000", "50k-70k", etc.
        """
        # Remove common prefixes/suffixes and extract numbers
        import re
        
        if not pay_text:
            return None, None
            
        # Find numeric ranges (could be 50k or $50,000)
        numbers = re.findall(r'\d+(?:,\d+)?', pay_text.replace(',', ''))
        
        if len(numbers) >= 2:
            # Simple approach - take first two numbers as min and max
            return int(numbers[0]), int(numbers[1])
        elif len(numbers) == 1:
            # If only one number, assume it's a min and set max to very high value
            return int(numbers[0]), 999999  # Arbitrary large number
            
        # If no numbers found, return None for both
        return None, None

    def clear_archetypes(self):
        """Clears all loaded archetypes."""
        self.archetypes = []

    