"""
LLM Classification and Reranking Engine for Job Analysis Pipeline

This module implements:
- Stage 6: Cheap LLM Classification (fast, structured fit analysis)
- Stage 7: Strong LLM Reranking (deep review of top candidates)
- Stage 8: Final Application Queue (combined scoring and ranking)
"""

import asyncio
import json
import os
import re
from typing import Dict, List, Optional
from openai import OpenAI
from anthropic import Anthropic
import google.genai as genai
from google.genai import types


# =====================================================
# STAGE 6: CHEAP LLM CLASSIFICATION PROMPTS
# =====================================================

CHEAP_LLM_SYSTEM_PROMPT = """You are a recruitment fit analyzer. Analyze the job description and candidate profile to produce a structured fit assessment.

RULES:
- Return ONLY valid JSON
- No explanations, no chain-of-thought
- Keep outputs compact
- Focus on factual fit analysis

OUTPUT FORMAT:
{
  "fit_score": <integer 0-100>,
  "decision": "<apply|maybe|skip>",
  "strengths": [<short string>, ...],
  "concerns": [<short string>, ...]
}"""

CHEAP_LLM_USER_TEMPLATE = """JOB DESCRIPTION:
{job_description}

JOB TITLE: {job_title}
JOB SKILLS: {job_skills}

CANDIDATE PROFILE:
{candidate_profile}

CANDIDATE SKILLS: {candidate_skills}

Analyze fit and return JSON."""


# =====================================================
# STAGE 7: STRONG LLM RERANKING PROMPTS
# =====================================================

STRONG_LLM_SYSTEM_PROMPT = """You are a senior recruitment consultant performing deep candidate-job fit analysis.

RULES:
- Return ONLY valid JSON
- No essays, no chain-of-thought
- Detect hidden red flags
- Evaluate actual fit quality
- Identify tailoring opportunities
- Identify likely recruiter bait
- Be critical and thorough

OUTPUT FORMAT:
{
  "final_score": <integer 0-100>,
  "priority": "<high|medium|low|skip>",
  "apply_recommendation": "<apply|maybe|skip>",
  "red_flags": [<short string>, ...],
  "tailoring_notes": [<short string>, ...],
  "recruiter_bait_likelihood": "<low|medium|high>",
  "detailed_fit_analysis": "<one sentence summary>"
}"""

STRONG_LLM_USER_TEMPLATE = """JOB DETAILS:
Title: {job_title}
Company: {company}
Description: {job_description}

Extracted Skills: {job_skills}
Requirements: {job_requirements}

Salary Range: {pay_range}
Work Type: {work_type}
Seniority: {seniority}

CANDIDATE PROFILE:
{candidate_profile}

Skills: {candidate_skills}

PREVIOUS ANALYSIS:
Semantic Score: {semantic_score}
Cheap LLM Fit Score: {cheap_llm_score}
Cheap LLM Decision: {cheap_llm_decision}
Strengths: {strengths}
Concerns: {concerns}

Perform deep analysis and return JSON."""


class CheapLLMClassifier:
    """
    Stage 6: Fast, structured fit analysis using lightweight models.
    Recommended: Gemini Flash-Lite, DeepSeek V3, or local models.
    """
    
    def __init__(self, provider: str = "gemini"):
        self.provider = provider
        self._init_client()
    
    def _init_client(self):
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)
            self.model = "gemini-2.0-flash-lite"
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o-mini"
        elif self.provider == "lm_studio":
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            self.model = "local-model"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def classify(self, job: Dict, candidate_profile: str, candidate_skills: List[str]) -> Dict:
        """
        Perform cheap LLM classification on a single job.
        Returns structured fit analysis.
        """
        job_description = job.get('features', {}).get('description', '')
        job_title = job.get('features', {}).get('title', '')
        job_skills = job.get('features', {}).get('skills', [])
        
        # Truncate if too long to save tokens
        if len(job_description) > 2000:
            job_description = job_description[:2000] + "..."
        
        prompt = CHEAP_LLM_USER_TEMPLATE.format(
            job_description=job_description,
            job_title=job_title,
            job_skills=", ".join(job_skills) if job_skills else "N/A",
            candidate_profile=candidate_profile[:1000],
            candidate_skills=", ".join(candidate_skills) if candidate_skills else "N/A"
        )
        
        try:
            if self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"{CHEAP_LLM_SYSTEM_PROMPT}\n\n{prompt}",
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                content = response.text
            elif self.provider in ("openai", "lm_studio"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": CHEAP_LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
            else:
                return self._default_result()
            
            if content is None:
                return self._default_result()
            
            # Parse JSON response
            result = json.loads(content)
            return self._validate_result(result)
            
        except Exception as e:
            print(f"[CheapLLMClassifier Error] {e}")
            return self._default_result()
    
    def _default_result(self) -> Dict:
        return {
            "fit_score": 50,
            "decision": "maybe",
            "strengths": [],
            "concerns": ["Analysis failed"]
        }
    
    def _validate_result(self, result: Dict) -> Dict:
        """Ensure result has required fields with valid types."""
        required = {
            "fit_score": (int, 50),
            "decision": (str, "maybe"),
            "strengths": (list, []),
            "concerns": (list, [])
        }
        
        validated = {}
        for key, (type_hint, default) in required.items():
            value = result.get(key, default)
            if not isinstance(value, type_hint):
                value = default
            validated[key] = value
        
        # Clamp fit_score to 0-100
        validated["fit_score"] = max(0, min(100, validated["fit_score"]))
        
        # Validate decision values
        if validated["decision"] not in ("apply", "maybe", "skip"):
            validated["decision"] = "maybe"
        
        return validated


class StrongLLMReranker:
    """
    Stage 7: Deep review of top candidates using powerful models.
    Recommended: DeepSeek V3, GPT-4, Claude Opus.
    """
    
    def __init__(self, provider: str = "claude"):
        self.provider = provider
        self._init_client()
    
    def _init_client(self):
        if self.provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-5-sonnet-20241022"
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o"
        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)
            self.model = "gemini-2.0-flash-exp"
        elif self.provider == "lm_studio":
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            self.model = "local-model"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def rerank(self, job: Dict, candidate_profile: str, candidate_skills: List[str], 
               cheap_result: Dict) -> Dict:
        """
        Perform deep analysis on a job that passed cheap classification.
        Returns detailed fit assessment with priority ranking.
        """
        features = job.get('features', {})
        job_description = features.get('description', '')
        job_title = features.get('title', '')
        job_skills = features.get('skills', [])
        job_requirements = features.get('requirements', [])
        pay_range = features.get('pay', 'Not specified')
        work_type = features.get('work_type', 'Unknown')
        seniority = features.get('seniority', 'Unknown')
        company = job.get('metadata', {}).get('source', 'Unknown')
        
        # Truncate for token efficiency
        if len(job_description) > 3000:
            job_description = job_description[:3000] + "..."
        
        prompt = STRONG_LLM_USER_TEMPLATE.format(
            job_title=job_title,
            company=company,
            job_description=job_description,
            job_skills=", ".join(job_skills) if job_skills else "N/A",
            job_requirements="\n".join(job_requirements[:10]) if job_requirements else "N/A",
            pay_range=pay_range,
            work_type=work_type,
            seniority=seniority,
            candidate_profile=candidate_profile[:1500],
            candidate_skills=", ".join(candidate_skills) if candidate_skills else "N/A",
            semantic_score=job.get('semantic_score', 0),
            cheap_llm_score=cheap_result.get('fit_score', 0),
            cheap_llm_decision=cheap_result.get('decision', 'unknown'),
            strengths=", ".join(cheap_result.get('strengths', [])),
            concerns=", ".join(cheap_result.get('concerns', []))
        )
        
        try:
            if self.provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    system=STRONG_LLM_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content
                if isinstance(content, list) and content:
                    first_block = content[0]
                    content = getattr(first_block, "text", None)
                    if content is None:
                        content = getattr(first_block, "output_text", None)
                    if content is None:
                        content = str(first_block)
            elif self.provider in ("openai", "lm_studio"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": STRONG_LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
            elif self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"{STRONG_LLM_SYSTEM_PROMPT}\n\n{prompt}",
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                content = response.text
            else:
                return self._default_result()
            
            if content is None:
                return self._default_result()
            
            result = json.loads(content)
            return self._validate_result(result)
            
        except Exception as e:
            print(f"[StrongLLMReranker Error] {e}")
            return self._default_result()
    
    def _default_result(self) -> Dict:
        return {
            "final_score": 50,
            "priority": "medium",
            "apply_recommendation": "maybe",
            "red_flags": ["Analysis failed"],
            "tailoring_notes": [],
            "recruiter_bait_likelihood": "medium",
            "detailed_fit_analysis": "Could not complete analysis"
        }
    
    def _validate_result(self, result: Dict) -> Dict:
        """Ensure result has required fields with valid types."""
        required = {
            "final_score": (int, 50),
            "priority": (str, "medium"),
            "apply_recommendation": (str, "maybe"),
            "red_flags": (list, []),
            "tailoring_notes": (list, []),
            "recruiter_bait_likelihood": (str, "medium"),
            "detailed_fit_analysis": (str, "")
        }
        
        validated = {}
        for key, (type_hint, default) in required.items():
            value = result.get(key, default)
            if not isinstance(value, type_hint):
                value = default
            validated[key] = value
        
        # Clamp final_score to 0-100
        validated["final_score"] = max(0, min(100, validated["final_score"]))
        
        # Validate enum values
        if validated["priority"] not in ("high", "medium", "low", "skip"):
            validated["priority"] = "medium"
        
        if validated["apply_recommendation"] not in ("apply", "maybe", "skip"):
            validated["apply_recommendation"] = "maybe"
        
        if validated["recruiter_bait_likelihood"] not in ("low", "medium", "high"):
            validated["recruiter_bait_likelihood"] = "medium"
        
        return validated


class FinalApplicationQueue:
    """
    Stage 8: Combine all scoring factors to produce final ranked application list.
    """
    
    def __init__(self):
        # Weights for final score calculation
        self.weights = {
            "semantic_score": 0.25,
            "cheap_llm_score": 0.20,
            "strong_llm_score": 0.35,
            "recency_bonus": 0.05,
            "salary_bonus": 0.05,
            "remote_bonus": 0.10
        }
    
    def calculate_final_score(self, job: Dict) -> float:
        """
        Calculate weighted final score combining all factors.
        """
        score = 0.0
        
        # Semantic score (Stage 5)
        semantic_score = job.get('semantic_score', 0) * 100  # Convert to 0-100
        score += self.weights["semantic_score"] * semantic_score
        
        # Cheap LLM score (Stage 6)
        cheap_score = job.get('cheap_llm_result', {}).get('fit_score', 50)
        score += self.weights["cheap_llm_score"] * cheap_score
        
        # Strong LLM score (Stage 7)
        strong_score = job.get('strong_llm_result', {}).get('final_score', 50)
        score += self.weights["strong_llm_score"] * strong_score
        
        # Recency bonus
        # Assume jobs within 7 days get full bonus, linear decay to 30 days
        days_old = job.get('days_old', 30)
        if days_old <= 7:
            recency_factor = 1.0
        elif days_old <= 30:
            recency_factor = max(0, (30 - days_old) / 23)
        else:
            recency_factor = 0
        score += self.weights["recency_bonus"] * recency_factor * 100
        
        # Salary bonus
        pay_range = job.get('features', {}).get('pay', '')
        salary_score = self._parse_salary_score(pay_range)
        score += self.weights["salary_bonus"] * salary_score * 100
        
        # Remote preference bonus
        work_type = job.get('features', {}).get('work_type', '').lower()
        is_remote = 'remote' in work_type
        score += self.weights["remote_bonus"] * (100 if is_remote else 50)
        
        return max(0, min(100, score))
    
    def _parse_salary_score(self, pay_range: str) -> float:
        """
        Parse salary range and return a normalized score (0-1).
        Assumes USD and typical tech salary ranges.
        Handles both "120k" and "120000" formats.
        """
        if not pay_range or pay_range == "Not specified":
            return 0.5  # Neutral if unknown
        
        # Check if using "k" notation
        has_k = 'k' in pay_range.lower()
        
        # Extract numbers from pay range
        numbers = re.findall(r'\d+(?:,\d+)?', pay_range.replace(',', ''))
        if not numbers:
            return 0.5
        
        # Use max of range if available
        values = [int(n) for n in numbers]
        max_salary = max(values)
        
        # Convert if using k notation
        if has_k:
            max_salary = max_salary * 1000
        
        # Normalize: assume 150k+ is excellent, <50k is poor
        if max_salary >= 150000:
            return 1.0
        elif max_salary <= 50000:
            return 0.2
        else:
            return 0.2 + (max_salary - 50000) / 100000 * 0.8
    
    def determine_priority(self, final_score: float, strong_llm_priority: str) -> str:
        """
        Determine application priority based on final score and LLM assessment.
        """
        if strong_llm_priority == "skip":
            return "skip"
        
        if final_score >= 80:
            return "high"
        elif final_score >= 65:
            return "medium"
        elif final_score >= 50:
            return "low"
        else:
            return "skip"
    
    def rank_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """
        Rank all jobs by final score and return sorted queue.
        """
        ranked_jobs = []
        
        for job in jobs:
            # Calculate final score
            final_score = self.calculate_final_score(job)
            job['final_score'] = final_score
            
            # Determine priority
            strong_priority = job.get('strong_llm_result', {}).get('priority', 'medium')
            priority = self.determine_priority(final_score, strong_priority)
            job['priority'] = priority
            
            # Get apply recommendation
            apply_rec = job.get('strong_llm_result', {}).get('apply_recommendation', 'maybe')
            job['apply_recommendation'] = apply_rec
            
            ranked_jobs.append(job)
        
        # Sort by final score descending
        ranked_jobs.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return ranked_jobs
    
    def generate_queue_summary(self, ranked_jobs: List[Dict]) -> Dict:
        """
        Generate summary statistics for the final queue.
        """
        summary = {
            "total_jobs": len(ranked_jobs),
            "high_priority": 0,
            "medium_priority": 0,
            "low_priority": 0,
            "skip": 0,
            "top_5_jobs": []
        }
        
        for job in ranked_jobs:
            priority = job.get('priority', 'medium')
            if priority == "high":
                summary["high_priority"] += 1
            elif priority == "medium":
                summary["medium_priority"] += 1
            elif priority == "low":
                summary["low_priority"] += 1
            else:
                summary["skip"] += 1
        
        # Top 5 jobs for display
        for job in ranked_jobs[:5]:
            summary["top_5_jobs"].append({
                "title": job.get('features', {}).get('title', 'Unknown'),
                "company": job.get('metadata', {}).get('source', 'Unknown'),
                "final_score": round(job.get('final_score', 0), 1),
                "priority": job.get('priority', 'unknown'),
                "semantic_score": round(job.get('semantic_score', 0) * 100, 1),
                "cheap_llm_score": job.get('cheap_llm_result', {}).get('fit_score', 0),
                "strong_llm_score": job.get('strong_llm_result', {}).get('final_score', 0)
            })
        
        return summary


async def process_stage_6(jobs: List[Dict], classifier: CheapLLMClassifier, 
                          candidate_profile: str, candidate_skills: List[str],
                          batch_size: int = 5) -> List[Dict]:
    """
    Process Stage 6: Cheap LLM Classification on filtered job pool.
    """
    print(f"Stage 6: Running cheap LLM classification on {len(jobs)} jobs...")
    
    for i, job in enumerate(jobs):
        print(f"  Processing job {i+1}/{len(jobs)}: {job.get('features', {}).get('title', 'Unknown')}")
        
        result = classifier.classify(job, candidate_profile, candidate_skills)
        job['cheap_llm_result'] = result
        
        # Rate limiting
        if (i + 1) % batch_size == 0:
            await asyncio.sleep(2)
    
    # Filter to only jobs with "apply" or "maybe" decisions
    shortlisted = [j for j in jobs if j.get('cheap_llm_result', {}).get('decision') in ('apply', 'maybe')]
    
    print(f"Stage 6 complete: {len(shortlisted)} jobs shortlisted from {len(jobs)}")
    return shortlisted


async def process_stage_7(jobs: List[Dict], reranker: StrongLLMReranker,
                          candidate_profile: str, candidate_skills: List[str],
                          top_n: int = 20) -> List[Dict]:
    """
    Process Stage 7: Strong LLM Reranking on top candidates.
    """
    # Sort by cheap LLM score and take top N
    jobs_sorted = sorted(jobs, key=lambda x: x.get('cheap_llm_result', {}).get('fit_score', 0), reverse=True)
    top_jobs = jobs_sorted[:top_n]
    
    print(f"Stage 7: Running strong LLM reranking on top {len(top_jobs)} jobs...")
    
    for i, job in enumerate(top_jobs):
        print(f"  Deep analysis {i+1}/{len(top_jobs)}: {job.get('features', {}).get('title', 'Unknown')}")
        
        cheap_result = job.get('cheap_llm_result', {})
        result = reranker.rerank(job, candidate_profile, candidate_skills, cheap_result)
        job['strong_llm_result'] = result
        
        # Rate limiting for expensive API calls
        await asyncio.sleep(3)
    
    print(f"Stage 7 complete: {len(top_jobs)} jobs deeply analyzed")
    return top_jobs


async def process_stage_8(jobs: List[Dict]) -> List[Dict]:
    """
    Process Stage 8: Generate final application queue.
    """
    print(f"Stage 8: Generating final application queue from {len(jobs)} jobs...")
    
    queue_generator = FinalApplicationQueue()
    ranked_jobs = queue_generator.rank_jobs(jobs)
    
    summary = queue_generator.generate_queue_summary(ranked_jobs)
    
    print(f"\n=== FINAL APPLICATION QUEUE SUMMARY ===")
    print(f"Total Jobs Analyzed: {summary['total_jobs']}")
    print(f"High Priority: {summary['high_priority']}")
    print(f"Medium Priority: {summary['medium_priority']}")
    print(f"Low Priority: {summary['low_priority']}")
    print(f"Skip: {summary['skip']}")
    
    print(f"\n=== TOP 5 JOBS TO APPLY ===")
    for i, job in enumerate(summary['top_5_jobs'], 1):
        print(f"{i}. {job['title']} at {job['company']}")
        print(f"   Final Score: {job['final_score']} | Priority: {job['priority']}")
        print(f"   Semantic: {job['semantic_score']}% | Cheap LLM: {job['cheap_llm_score']} | Strong LLM: {job['strong_llm_score']}")
    
    return ranked_jobs