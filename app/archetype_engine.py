import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from app.vector_engine import VectorEngine, LocalEmbeddingProvider

@dataclass
class Archetype:
    """Represents a profile (e.g., Resume, User Profile) used for comparison."""
    name: str
    type: str  # e.g., 'resume', 'user_profile'
    title_embedding: Optional[np.ndarray] = None
    requirements_embedding: Optional[np.ndarray] = None
    responsibilities_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_combined_embedding(self) -> Optional[np.ndarray]:
        """Returns a combined embedding from available archetype vectors."""
        embeddings = [
            emb for emb in (
                self.title_embedding,
                self.requirements_embedding,
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
        self.negative_archetypes: List[Archetype] = []

    def load_archetype(self, name: str, archetype_data: Dict[str, str], archetype_type: str = "benchmark", metadata: Optional[Dict] = None) -> Archetype:
        """
        Loads an archetype by generating the three required embeddings defined in Stage 4.
        archetype_data keys: 'title', 'requirements', 'responsibilities'
        """
        title_text = archetype_data.get("title", "")
        requirements_text = archetype_data.get("requirements", "") or archetype_data.get("skills", "")
        resp_text = archetype_data.get("responsibilities", "")

        # In a production scenario, you would check a cache/database here 
        # before calling get_embeddings to satisfy the "must be cached" constraint.
        
        embeddings = self.vector_engine.get_embeddings([title_text, requirements_text, resp_text])
        
        new_archetype = Archetype(
            name=name,
            type=archetype_type,
            title_embedding=embeddings[0],
            requirements_embedding=embeddings[1],
            responsibilities_embedding=embeddings[2],
            metadata=metadata or {}
        )
        self.archetypes.append(new_archetype)
        return new_archetype

    def add_archetype(self, archetype: Archetype):
        """Adds a pre-loaded/cached archetype directly to the manager."""
        self.archetypes.append(archetype)

    def add_negative_archetype(self, archetype: Archetype):
        """Adds a pre-loaded/cached negative archetype (anti-profile) directly to the manager."""
        self.negative_archetypes.append(archetype)

    def create_semantic_document(self, job_data: Dict[str, Any]) -> str:
        """
        Creates a single semantic document string from job data for embedding.
        Combines title, description, and other key fields to capture the role's essence.
        """
        title = job_data.get("title", "")
        description = job_data.get("description", "")
        # Add other relevant fields if they exist in the job_data
        requirements = " ".join(job_data.get("requirements", [])) if isinstance(job_data.get("requirements"), list) else ""
        
        # Construct a cohesive text block for the embedding model
        document = f"Job Title: {title}. Description: {description} Requirements: {requirements}"
        return document

    def _get_job_embedding(self, job_data: Dict[str, Any], embedding_key: str, text: str) -> Optional[np.ndarray]:
        """
        Helper to retrieve a pre-computed job embedding from the job dict,
        falling back to generating a new embedding from text if not available.

        Args:
            job_data: The full job dictionary.
            embedding_key: Key in job_data['embeddings'] (e.g. 'title_vector').
            text: Fallback text to embed if no pre-computed embedding exists.

        Returns:
            A numpy array embedding, or None if neither source is available.
        """
        # 1. Try pre-computed embedding from the job dict
        pre_computed = job_data.get("embeddings", {}).get(embedding_key)
        if pre_computed is not None:
            # Convert Python list → numpy array if needed (e.g. loaded from DB)
            if isinstance(pre_computed, list):
                if len(pre_computed) == 0:
                    return None  # empty list is not a valid embedding
                return np.array(pre_computed, dtype=np.float32)
            if isinstance(pre_computed, np.ndarray):
                if pre_computed.size == 0:
                    return None  # empty array is not a valid embedding
                return pre_computed

        # 2. Fall back to generating a fresh embedding from text
        if text:
            return self.vector_engine.get_embeddings([text])[0]

        return None

    def compare_job_to_archetypes(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compares a single job against all loaded archetypes.
        Returns a list of matches with detailed similarity scores (title, skills, responsibilities)
        and metadata. This enables the weighted scoring formula in Stage 5D.

        Uses pre-computed embeddings from the job dict when available (Stage 2 output)
        to avoid redundant embedding generation and ensure consistency with the
        pipeline's configured embedding provider.
        """
        if not self.archetypes:
            return []

        # 1. Extract job components for individual similarity calculations
        job_title = job_data.get("features", {}).get("title", "")
        job_requirements = job_data.get("features", {}).get("requirements", [])
        job_responsibilities = job_data.get("features", {}).get("responsibilities", [])
        job_description = job_data.get("features", {}).get("description", "")

        # Use responsibilities if available, otherwise fall back to description
        job_responsibilities_text = "\n".join(job_responsibilities) if job_responsibilities else job_description

        # Prepare fallback text strings for embedding generation when no pre-computed ones exist
        requirements_text = "\n".join(job_requirements) if job_requirements else ""

        # 2. Retrieve or generate embeddings for each job component
        #    Priority: pre-computed embeddings > fresh generation from text
        job_title_embedding = self._get_job_embedding(job_data, "title_vector", job_title)
        job_requirements_embedding = self._get_job_embedding(job_data, "requirements_vector", requirements_text)
        job_responsibilities_embedding = self._get_job_embedding(
            job_data, "responsibilities_vector", job_responsibilities_text
        )

        matches = []
        for archetype in self.archetypes:
            # Calculate individual similarity scores for each component
            title_similarity = 0.0
            requirements_similarity = 0.0
            responsibilities_similarity = 0.0

            if job_title_embedding is not None and archetype.title_embedding is not None:
                title_similarity = self.vector_engine.compute_similarity(job_title_embedding, archetype.title_embedding)

            if job_requirements_embedding is not None and archetype.requirements_embedding is not None:
                requirements_similarity = self.vector_engine.compute_similarity(job_requirements_embedding, archetype.requirements_embedding)

            if job_responsibilities_embedding is not None and archetype.responsibilities_embedding is not None:
                responsibilities_similarity = self.vector_engine.compute_similarity(
                    job_responsibilities_embedding, archetype.responsibilities_embedding
                )

            # Calculate combined similarity for sorting (simple average as fallback)
            combined_score = (title_similarity + requirements_similarity + responsibilities_similarity) / 3.0

            matches.append({
                "archetype_name": archetype.name,
                "archetype_type": archetype.type,
                "similarity_score": float(combined_score),  # For backward compatibility
                "title_similarity": float(title_similarity),
                "requirements_similarity": float(requirements_similarity),
                "responsibility_similarity": float(responsibilities_similarity),
                "metadata": archetype.metadata
            })

        # Sort matches by highest combined score first
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

    def compare_job_to_negative_archetypes(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares a job against all loaded negative archetypes (anti-profiles).
        Returns a dict with 'max_negative_similarity', 'worst_match_name', and detailed matches list.
        """
        if not self.negative_archetypes:
            return {
                "max_negative_similarity": 0.0,
                "worst_match_name": "",
                "matches": []
            }

        job_title = job_data.get("features", {}).get("title", "")
        job_requirements = job_data.get("features", {}).get("requirements", [])
        job_responsibilities = job_data.get("features", {}).get("responsibilities", [])
        job_description = job_data.get("features", {}).get("description", "")
        job_resp_text = "\n".join(job_responsibilities) if job_responsibilities else job_description
        req_text = "\n".join(job_requirements) if job_requirements else ""

        job_title_emb = self._get_job_embedding(job_data, "title_vector", job_title)
        job_req_emb = self._get_job_embedding(job_data, "requirements_vector", req_text)
        job_resp_emb = self._get_job_embedding(job_data, "responsibilities_vector", job_resp_text)

        matches = []
        for neg_arch in self.negative_archetypes:
            title_sim = 0.0
            req_sim = 0.0
            resp_sim = 0.0

            if job_title_emb is not None and neg_arch.title_embedding is not None:
                title_sim = self.vector_engine.compute_similarity(job_title_emb, neg_arch.title_embedding)

            if job_req_emb is not None and neg_arch.requirements_embedding is not None:
                req_sim = self.vector_engine.compute_similarity(job_req_emb, neg_arch.requirements_embedding)

            if job_resp_emb is not None and neg_arch.responsibilities_embedding is not None:
                resp_sim = self.vector_engine.compute_similarity(job_resp_emb, neg_arch.responsibilities_embedding)

            # Title similarity heavily signals role alignment; responsibilities signal duties
            combined_neg_sim = max(
                title_sim,
                0.60 * title_sim + 0.40 * max(req_sim, resp_sim)
            )

            matches.append({
                "archetype_name": neg_arch.name,
                "archetype_type": neg_arch.type,
                "negative_similarity": float(combined_neg_sim),
                "title_similarity": float(title_sim),
                "requirements_similarity": float(req_sim),
                "responsibility_similarity": float(resp_sim)
            })

        matches.sort(key=lambda x: x["negative_similarity"], reverse=True)
        worst_match = matches[0] if matches else None

        return {
            "max_negative_similarity": worst_match["negative_similarity"] if worst_match else 0.0,
            "worst_match_name": worst_match["archetype_name"] if worst_match else "",
            "matches": matches
        }

    def clear_archetypes(self):
        """Clears all loaded positive and negative archetypes."""
        self.archetypes = []
        self.negative_archetypes = []


    