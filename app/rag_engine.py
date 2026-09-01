import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from app.pull_data import DataPuller
from app.ai_engine import AIEngine
from app.vector_engine import VectorEngine, CallableEmbeddingProvider
from app.ai_utils import generate_embeddings
from app.config_utils import _load_user_config

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    RAG Retrieval & Generation Engine built on top of pgvector, 
    VectorEngine, and AIEngine.
    """
    def __init__(self, dp: Optional[DataPuller] = None, 
                 ai_engine: Optional[AIEngine] = None, 
                 vector_engine: Optional[VectorEngine] = None):
        self.dp = dp
        self.ai = ai_engine or AIEngine(default_provider_name="lm_studio")
        
        user_config = _load_user_config()
        self.embeddings_llm = user_config.get("embeddings_llm", os.getenv("EMBEDDINGS_LLM", "lm_studio"))
        self.embeddings_model = user_config.get("embeddings_model", os.getenv("EMBEDDINGS_MODEL", "local-model"))

        if vector_engine is None:
            embed_fn = lambda text: generate_embeddings(self.ai, text, provider_name=self.embeddings_llm)
            provider = CallableEmbeddingProvider(embed_fn)
            self.vector_engine = VectorEngine(provider=provider)
        else:
            self.vector_engine = vector_engine

        self.validated_consistency = False
        self.current_dim = None

    def validate_embedding_consistency(self) -> Tuple[bool, str]:
        """
        Verify that active embedding provider's output dimension matches 
        stored DB metadata. Stores or updates rag_metadata on model change.
        """
        if not self.dp:
            return True, "No database connection available to validate."

        try:
            sample_emb = generate_embeddings(self.ai, "consistency check", provider_name=self.embeddings_llm)
            if not sample_emb:
                return False, "Could not generate sample embedding for consistency check."
            
            self.current_dim = len(sample_emb)
            model_key = f"{self.embeddings_llm}:{self.embeddings_model}"

            # Query existing metadata
            rows = self.dp.conn.execute_sql(
                "SELECT value_text FROM rag_metadata WHERE key_name = %s",
                ("embeddings_model_config",),
                fetch=True
            )

            if rows and rows[0]:
                db_config_str = rows[0].get("value_text") if isinstance(rows[0], dict) else rows[0][0]
                try:
                    db_config = json.loads(db_config_str)
                    db_dim = db_config.get("dimension")
                    db_model = db_config.get("model")

                    if db_dim and db_dim != self.current_dim:
                        msg = (
                            f"WARNING: Dimension mismatch detected! Active model '{model_key}' "
                            f"outputs {self.current_dim}d, but database was indexed with {db_dim}d ({db_model})."
                        )
                        logger.warning(msg)
                        return False, msg
                except Exception:
                    pass

            # Upsert current configuration metadata
            meta_value = json.dumps({"model": model_key, "dimension": self.current_dim})
            upsert_sql = """
            INSERT INTO rag_metadata (key_name, value_text, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (key_name) 
            DO UPDATE SET value_text = EXCLUDED.value_text, updated_at = CURRENT_TIMESTAMP;
            """
            self.dp.conn.execute_sql(upsert_sql, ("embeddings_model_config", meta_value))
            self.validated_consistency = True
            return True, f"Embedding model consistency validated ({self.current_dim}d)."

        except Exception as e:
            return False, f"Embedding consistency check failed: {e}"

    def index_job(self, job: Dict[str, Any]) -> bool:
        """
        Extract typed chunks from job features and LLM evaluations, 
        generate embeddings, and save to rag_documents table.
        """
        if not self.dp:
            return False

        metadata_dict = job.get('metadata', {})
        features_dict = job.get('features', {})
        job_id = metadata_dict.get('job_id') or job.get('id')

        if not job_id:
            return False

        company_name = metadata_dict.get('company_name', '')
        title = features_dict.get('title', '')
        work_type = features_dict.get('work_type', '')
        pay_range = features_dict.get('pay', '')
        seniority = features_dict.get('seniority', '')
        final_score = job.get('final_score') or job.get('strong_llm_result', {}).get('final_score')
        priority = job.get('priority') or job.get('strong_llm_result', {}).get('priority')

        meta_payload = {
            "title": title,
            "company_name": company_name,
            "work_type": work_type,
            "pay_range": pay_range,
            "seniority": seniority,
            "final_score": final_score,
            "priority": priority,
            "link": metadata_dict.get('link', '')
        }

        chunks_to_index = []

        # 1. Summary Chunk
        summary = features_dict.get('summary', '') or features_dict.get('description', '')
        if summary and len(summary.strip()) > 30:
            chunks_to_index.append(('summary', summary.strip()))

        # 2. Requirements Chunk
        reqs = features_dict.get('requirements', [])
        if reqs:
            reqs_text = "Required Skills & Qualifications:\n" + "\n".join(f"- {r}" for r in reqs)
            chunks_to_index.append(('requirements', reqs_text))

        # 3. Responsibilities Chunk
        resps = features_dict.get('responsibilities', [])
        if resps:
            resps_text = "Core Responsibilities:\n" + "\n".join(f"- {r}" for r in resps)
            chunks_to_index.append(('responsibilities', resps_text))

        # 4. Pay Chunk
        if pay_range and len(pay_range.strip()) > 0:
            chunks_to_index.append(('pay', f"Pay Range: {pay_range.strip()}"))

        # 5. Location & Work Arrangement Chunk
        loc_parts = []
        if features_dict.get('location'):
            loc_parts.append(f"Location: {features_dict.get('location')}")
        elif features_dict.get('city') or features_dict.get('state'):
            city_state = ", ".join(filter(None, [features_dict.get('city'), features_dict.get('state')]))
            loc_parts.append(f"Location: {city_state}")
        if work_type and work_type not in ('Unknown', 'NA'):
            loc_parts.append(f"Work Arrangement: {work_type}")
        if features_dict.get('timezone') and features_dict.get('timezone') != 'NA':
            loc_parts.append(f"Timezone: {features_dict.get('timezone')}")
        
        if loc_parts:
            chunks_to_index.append(('location', ". ".join(loc_parts)))

        # 6. LLM Reasoning Chunks (Cheap & Strong LLM results)
        cheap_res = job.get('cheap_llm_result', {})
        strong_res = job.get('strong_llm_result', {})
        
        reasoning_parts = []
        if cheap_res.get('strengths'):
            reasoning_parts.append("Key Strengths: " + ", ".join(cheap_res.get('strengths', [])))
        if cheap_res.get('concerns'):
            reasoning_parts.append("Concerns: " + ", ".join(cheap_res.get('concerns', [])))
        if strong_res.get('red_flags'):
            reasoning_parts.append("Red Flags: " + ", ".join(strong_res.get('red_flags', [])))
        if strong_res.get('tailoring_notes'):
            reasoning_parts.append("Tailoring Guidance: " + ", ".join(strong_res.get('tailoring_notes', [])))
        if strong_res.get('driving_points'):
            reasoning_parts.append("Key Driving Points: " + ", ".join(strong_res.get('driving_points', [])))
        if strong_res.get('detailed_fit_analysis'):
            reasoning_parts.append("Detailed Fit Analysis:\n" + str(strong_res.get('detailed_fit_analysis')))

        if reasoning_parts:
            chunks_to_index.append(('llm_reasoning', "\n".join(reasoning_parts)))

        if not chunks_to_index:
            return False

        # Clear old chunks for this job_id to prevent duplicates
        try:
            self.dp.conn.execute_sql("DELETE FROM rag_documents WHERE job_id = %s", (job_id,))
        except Exception:
            pass

        existing_embeddings = job.get('embeddings', {}) or {}

        # Generate embeddings & insert (reusing pre-computed job vectors when available)
        for chunk_type, content in chunks_to_index:
            emb = None
            if chunk_type == 'summary' and existing_embeddings.get('description_vector'):
                emb = existing_embeddings['description_vector']
            elif chunk_type == 'requirements' and existing_embeddings.get('requirements_vector'):
                emb = existing_embeddings['requirements_vector']
            elif chunk_type == 'responsibilities' and existing_embeddings.get('responsibilities_vector'):
                emb = existing_embeddings['responsibilities_vector']
            elif chunk_type == 'pay' and existing_embeddings.get('pay_vector'):
                emb = existing_embeddings['pay_vector']
            elif chunk_type == 'location' and existing_embeddings.get('location_vector'):
                emb = existing_embeddings['location_vector']
            else:
                emb = generate_embeddings(self.ai, content, provider_name=self.embeddings_llm)

            emb_str = f"[{','.join(str(x) for x in emb)}]" if emb else None

            insert_sql = """
            INSERT INTO rag_documents (job_id, chunk_type, content, embedding, metadata)
            VALUES (%s, %s, %s, %s::vector, %s::jsonb);
            """
            try:
                self.dp.conn.execute_sql(insert_sql, (job_id, chunk_type, content, emb_str, json.dumps(meta_payload, default=str)))
            except Exception as e:
                # Fallback if vector cast fails
                fallback_sql = """
                INSERT INTO rag_documents (job_id, chunk_type, content, metadata)
                VALUES (%s, %s, %s, %s::jsonb);
                """
                self.dp.conn.execute_sql(fallback_sql, (job_id, chunk_type, content, json.dumps(meta_payload, default=str)))

        return True

    def index_candidate_profile(self, profile_text: str, source_label: str = "candidate_profile") -> bool:
        """
        Chunk and index candidate resume or profile text into rag_documents.
        """
        if not self.dp or not profile_text:
            return False

        chunks = self.vector_engine.chunk_text(profile_text, chunk_size=300, overlap=40)
        
        # Clear existing profile chunks for source_label
        try:
            self.dp.conn.execute_sql("DELETE FROM rag_documents WHERE job_id IS NULL AND chunk_type = %s", (source_label,))
        except Exception:
            pass

        for idx, chunk in enumerate(chunks):
            emb = generate_embeddings(self.ai, chunk, provider_name=self.embeddings_llm)
            emb_str = f"[{','.join(str(x) for x in emb)}]" if emb else None
            meta_payload = {"source": source_label, "chunk_index": idx}

            insert_sql = """
            INSERT INTO rag_documents (job_id, chunk_type, content, embedding, metadata)
            VALUES (NULL, %s, %s, %s::vector, %s::jsonb);
            """
            try:
                self.dp.conn.execute_sql(insert_sql, (source_label, chunk, emb_str, json.dumps(meta_payload, default=str)))
            except Exception:
                fallback_sql = """
                INSERT INTO rag_documents (job_id, chunk_type, content, metadata)
                VALUES (NULL, %s, %s, %s::jsonb);
                """
                self.dp.conn.execute_sql(fallback_sql, (source_label, chunk, json.dumps(meta_payload, default=str)))

        return True

    def hybrid_search(self, query_text: str, 
                      work_type: Optional[str] = None,
                      min_pay: Optional[float] = None,
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval (pgvector cosine distance + metadata filters).
        """
        if not self.dp:
            return []

        query_emb = generate_embeddings(self.ai, query_text, provider_name=self.embeddings_llm)
        emb_str = f"[{','.join(str(x) for x in query_emb)}]" if query_emb else None

        where_clauses = ["rd.content IS NOT NULL"]
        params = []

        if emb_str:
            params.append(emb_str)

        if work_type:
            where_clauses.append("(rd.metadata->>'work_type') ILIKE %s")
            params.append(f"%{work_type}%")

        where_sql = " AND ".join(where_clauses)

        if emb_str:
            sql_query = f"""
            SELECT 
                rd.id, rd.job_id, rd.chunk_type, rd.content, rd.metadata,
                (rd.embedding <=> %s::vector) AS distance,
                j.job_name, c.company_name, j.link
            FROM rag_documents rd
            LEFT JOIN job j ON rd.job_id = j.id
            LEFT JOIN company c ON j.company_id = c.id
            WHERE {where_sql}
            ORDER BY distance ASC
            LIMIT %s;
            """
            params.append(top_k)
        else:
            # Text matching fallback if embeddings unavailable
            params.append(f"%{query_text}%")
            sql_query = f"""
            SELECT 
                rd.id, rd.job_id, rd.chunk_type, rd.content, rd.metadata,
                0.5::float AS distance,
                j.job_name, c.company_name, j.link
            FROM rag_documents rd
            LEFT JOIN job j ON rd.job_id = j.id
            LEFT JOIN company c ON j.company_id = c.id
            WHERE {where_sql} AND rd.content ILIKE %s
            LIMIT %s;
            """
            params.append(top_k)

        try:
            rows = self.dp.conn.execute_sql(sql_query, tuple(params), fetch=True) or []
            results = []
            for r in rows:
                if isinstance(r, dict):
                    row_dict = dict(r)
                else:
                    row_dict = {
                        "id": r[0], "job_id": r[1], "chunk_type": r[2], "content": r[3],
                        "metadata": r[4], "distance": r[5], "job_name": r[6],
                        "company_name": r[7], "link": r[8]
                    }
                if row_dict.get("distance") is not None:
                    try:
                        row_dict["distance"] = float(row_dict["distance"])
                    except (ValueError, TypeError):
                        pass
                row_dict["similarity"] = round(1.0 - float(row_dict.get("distance", 0.5)), 4)
                results.append(row_dict)
            return results
        except Exception as e:
            logger.error(f"Hybrid search SQL failed: {e}")
            return []

    async def generate_tailored_application(self, job_id: int) -> Dict[str, Any]:
        """
        Synthesize job requirements, Stage 7 tailoring notes, and 
        matching candidate profile chunks into a tailored application strategy.
        """
        if not self.dp:
            return {"error": "Database connection unavailable."}

        # Fetch job extractions & LLM evaluation
        job_rows = self.dp.conn.execute_sql(
            """
            SELECT j.id, j.job_name, c.company_name, j.description, j.requirements, j.responsibilities,
                   slr.tailoring_notes, slr.driving_points, slr.detailed_fit_analysis
            FROM job j
            LEFT JOIN company c ON j.company_id = c.id
            LEFT JOIN strong_llm_results slr ON j.id = slr.job_id
            WHERE j.id = %s;
            """,
            (job_id,),
            fetch=True
        )

        if not job_rows:
            return {"error": f"Job ID {job_id} not found."}

        job_data = dict(job_rows[0]) if isinstance(job_rows[0], dict) else {
            "job_name": job_rows[0][1], "company_name": job_rows[0][2],
            "requirements": job_rows[0][4], "responsibilities": job_rows[0][5],
            "tailoring_notes": job_rows[0][6], "driving_points": job_rows[0][7],
            "detailed_fit_analysis": job_rows[0][8]
        }

        # Retrieve relevant candidate profile chunks
        reqs_str = str(job_data.get("requirements", ""))
        matching_chunks = self.hybrid_search(reqs_str, top_k=3)
        candidate_context = "\n---\n".join(c["content"] for c in matching_chunks if c.get("content"))

        prompt = f"""
        You are an expert career consultant. Generate a targeted job application strategy for:
        
        JOB TITLE: {job_data.get('job_name')}
        COMPANY: {job_data.get('company_name')}
        
        REQUIREMENTS:
        {job_data.get('requirements')}
        
        STAGE 7 TAILORING NOTES:
        {job_data.get('tailoring_notes')}
        
        KEY DRIVING POINTS:
        {job_data.get('driving_points')}
        
        MATCHING CANDIDATE EXPERIENCE CHUNKS:
        {candidate_context}
        
        Output a structured JSON response with keys:
        1. "cover_letter_hook": A strong custom opening sentence.
        2. "resume_bullet_tailoring": List of 3 bullet point modifications highlighting relevant achievements.
        3. "interview_talking_points": List of 3 key stories or technical highlights to emphasize.
        """
        response_text = await self.ai.call_llm(prompt)
        return {
            "job_id": job_id,
            "job_name": job_data.get("job_name"),
            "company_name": job_data.get("company_name"),
            "tailoring_strategy": response_text,
            "matching_candidate_chunks": [c.get("content", "") for c in matching_chunks if isinstance(c, dict) and c.get("content")]
        }

    async def query_job_corpus(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer natural language queries across the job corpus with cited job links.
        """
        retrieved_docs = self.hybrid_search(user_query, top_k=top_k)
        
        context_blocks = []
        for i, doc in enumerate(retrieved_docs, 1):
            block = (
                f"[{i}] Job: {doc.get('job_name', 'Unknown')} at {doc.get('company_name', 'Unknown')}\n"
                f"Link: {doc.get('link', 'N/A')}\n"
                f"Chunk Type: {doc.get('chunk_type')}\n"
                f"Content: {doc.get('content')}"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        prompt = f"""
        Answer the user's question using ONLY the provided job context blocks. 
        Cite the job title, company, and link for any specific role mentioned.
        
        USER QUESTION:
        {user_query}
        
        RETRIEVED JOB CONTEXT:
        {context_str}
        """

        answer = await self.ai.call_llm(prompt)
        return {
            "query": user_query,
            "answer": answer,
            "sources": [
                {
                    "job_name": d.get("job_name"),
                    "company_name": d.get("company_name"),
                    "link": d.get("link"),
                    "similarity": d.get("similarity")
                }
                for d in retrieved_docs
            ]
        }
