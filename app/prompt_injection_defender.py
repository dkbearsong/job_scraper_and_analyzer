"""
Prompt Injection Defense Module for Job Scraper and Analyzer.

Provides:
- Deterministic pre-sanitization of untrusted external text (job descriptions, titles, etc.)
- Architectural tag wrapping for LLM prompt isolation
- Post-validation and sanitization of LLM-extracted data before database storage
"""

import re
from typing import Dict, Any, List, Union

# Zero-width and invisible unicode characters pattern
ZERO_WIDTH_PATTERN = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060\u202a-\u202e\u2000-\u200a]')

# HTML comments pattern
HTML_COMMENT_PATTERN = re.compile(r'<!--[\s\S]*?-->')

# Role boundary injection patterns
ROLE_INJECTION_PATTERN = re.compile(
    r'(?i)(\n|\r|^)\s*(System|User|Assistant|Human|AI|Instruction|Prompt)\s*:\s*',
    re.MULTILINE
)

# Known prompt injection directive patterns
INJECTION_DIRECTIVES = [
    re.compile(r'(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b'),
    re.compile(r'(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions?\b'),
    re.compile(r'(?i)\bforget\s+(all\s+)?(previous|prior|above)\s+instructions?\b'),
    re.compile(r'(?i)\byou\s+are\s+(now\s+)?an?\s+AI\s+assistant\b'),
    re.compile(r'(?i)\bnew\s+(system\s+)?instruction:\b'),
    re.compile(r'(?i)\boverride\s+(the\s+)?(system\s+)?prompt\b'),
    re.compile(r'(?i)\binclude\s+the\s+(following\s+)?(phrase|word|keyphrase|code)\b.*?\bin\s+(the\s+)?(cover\s+letter|resume|application)\b'),
    re.compile(r'(?i)\badd\s+signs?\s+into\s+(resumes?|cover\s+letters?)\b'),
    re.compile(r'(?i)\bset\s+up\s+with\s+AI\b'),
    re.compile(r'(?i)<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<\|endoftext\|>'),
]

def sanitize_untrusted_text(text: str) -> str:
    """
    Deterministically cleans raw untrusted text (e.g. scraped job descriptions, titles)
    by removing zero-width characters, HTML comments, role injection tokens, and neutralizing
    known prompt injection commands.
    """
    if not text or not isinstance(text, str):
        return text or ""

    # 1. Strip zero-width and invisible unicode characters
    cleaned = ZERO_WIDTH_PATTERN.sub('', text)

    # 2. Strip HTML comments
    cleaned = HTML_COMMENT_PATTERN.sub('', cleaned)

    # 3. Neutralize role markers (e.g., "System:", "\nUser:") to prevent role switching
    cleaned = ROLE_INJECTION_PATTERN.sub(r'\1[Filtered Header]: ', cleaned)

    # 4. Neutralize prompt injection directive phrases
    for pattern in INJECTION_DIRECTIVES:
        cleaned = pattern.sub('[filtered prompt injection directive]', cleaned)

    return cleaned.strip()


def wrap_untrusted_content(tag_name: str, content: str) -> str:
    """
    Wraps untrusted content within explicit XML-style boundary tags for prompt isolation.
    Sanitizes any closing tags within the content to prevent tag breakout attacks.
    """
    if not content:
        content = ""
    safe_tag = re.sub(r'[^a-zA-Z0-9_]', '_', tag_name)
    # Neutralize any accidental or intentional closing tag inside the content
    closing_tag_pattern = re.compile(rf'</\s*{re.escape(safe_tag)}\s*>', re.IGNORECASE)
    safe_content = closing_tag_pattern.sub(f'[escaped_tag]', content)
    return f"<{safe_tag}>\n{safe_content}\n</{safe_tag}>"


def is_suspicious_text(text: str) -> bool:
    """
    Checks if a string contains prompt injection triggers or suspicious command phrases.
    """
    if not text or not isinstance(text, str):
        return False
    
    for pattern in INJECTION_DIRECTIVES:
        if pattern.search(text):
            return True
            
    if re.search(r'(?i)\b(ignore|disregard|override)\b.*?\binstructions?\b', text):
        return True
    if re.search(r'(?i)\b(cover\s+letter|resume)\b.*?\b(banana|ai\s+generated|bot|trigger)\b', text):
        return True
        
    return False


def validate_and_clean_extracted_data(data: Any) -> Any:
    """
    Post-validation step before saving LLM extraction / scoring outputs into the database.
    Recursively inspects extracted dictionaries and lists to remove residual prompt injection
    payloads or suspicious injected items.
    """
    if isinstance(data, dict):
        cleaned_dict = {}
        for key, val in data.items():
            if isinstance(val, str):
                if is_suspicious_text(val):
                    # Replace suspicious string output with sanitized placeholder or clean string
                    cleaned_dict[key] = "[sanitized suspicious output]"
                else:
                    cleaned_dict[key] = val
            elif isinstance(val, list):
                cleaned_list = []
                for item in val:
                    if isinstance(item, str):
                        if not is_suspicious_text(item):
                            cleaned_list.append(item)
                    else:
                        cleaned_list.append(validate_and_clean_extracted_data(item))
                cleaned_dict[key] = cleaned_list
            else:
                cleaned_dict[key] = validate_and_clean_extracted_data(val)
        return cleaned_dict
    elif isinstance(data, list):
        cleaned_list = []
        for item in data:
            if isinstance(item, str):
                if not is_suspicious_text(item):
                    cleaned_list.append(item)
            else:
                cleaned_list.append(validate_and_clean_extracted_data(item))
        return cleaned_list
    else:
        return data
