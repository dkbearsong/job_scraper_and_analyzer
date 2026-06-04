# app/text_engine.py

import re

class TextProcessor:
    """
    A reusable engine for text analysis including regex, 
    keyword matching, parsing, and deterministic classification.
    """

    def __init__(self, default_case_sensitive=False):
        self.case_sensitive = default_case_sensitive

    def _prepare_text(self, text: str) -> str:
        """Internal helper to handle case sensitivity."""
        if not self.case_sensitive:
            return text.lower()
        return text

    def regex_search(self, text: str, pattern: str) -> list:
        """Uses regular expressions to find all matches."""
        processed_text = self._prepare_text(text)
        search_pattern = pattern if self.case_sensitive else pattern.lower()
        return re.findall(search_pattern, processed_text)

    def keyword_match(self, text: str, keywords: list, exact_match: bool = False) -> dict:
        """Checks for the presence of specific keywords."""
        processed_text = self._prepare_text(text)
        found_keywords = []

        for kw in keywords:
            kw_proc = kw if self.case_sensitive else kw.lower()
            if exact_match:
                if re.search(rf'\b{re.escape(kw_proc)}\b', processed_text):
                    found_keywords.append(kw)
            else:
                if kw_proc in processed_text:
                    found_keywords.append(kw)
        
        return {
            "match_found": len(found_keywords) > 0,
            "matches": found_keywords
        }

    def get_section_content(self, text: str, header_name: str) -> str:
        """Extracts a block of text belonging to a specific header."""
        pattern = rf"{re.escape(header_name)}[:\n\r]+([\s\S]*?)(?=\n[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*[:\n]|\Z)"
        match = re.search(pattern, text, re.IGNORECASE if not self.case_sensitive else 0)
        return match.group(1).strip() if match else ""

    def clean_list_from_text(self, text: str) -> list:
        """Converts a block of text into a clean list of strings."""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            clean_line = re.sub(r'^[\s\-\*\•\d\.\)]+', '', line).strip()
            if clean_line:
                cleaned.append(clean_line)
        return cleaned


    def detect_work_type(self, text: str) -> str:
        """Identifies Remote, Hybrid, or Onsite via keyword matching."""
        text = self._prepare_text(text)
        if re.search(r'\bremote\b|\bwork from home\b|\bfreely remote\b', text):
            return "Remote"
        if re.search(r'\bhybrid\b', text):
            return "Hybrid"
        if re.search(r'\bonsite\b|\boffice\b', text):
            return "Onsite"
        return "Unknown"

    def detect_seniority(self, text: str) -> str:
        """Identifies seniority level via keyword matching."""
        text = self._prepare_text(text)
        # Order matters: check for higher levels first
        rules = {
            "C-Suite": r'c-suite|executive|vp|vice president|chief officer',
            "Management": r'manager|director|head of|lead',
            "Senior": r'senior|sr\.|principal|staff',
            "Mid-Level": r'intermediate|mid-level|specialist',
            "Junior": r'junior|jr\.|entry level|associate|intern'
        }
        for level, pattern in rules.items():
            if re.search(pattern, text):
                return level
        return "Entry/Unknown"

    def extract_salary(self, text: str) -> str:
        """Attempts to find a salary range in the text."""
        # This is a basic regex for patterns like "$50,000 - $70,000" or "$50k - $70k"
        pattern = r'(\$\d{1,3}(?:,\d{3})*(?:\s?[kK])?\s?[-–—to]+\s?\$\d{1,3}(?:,\d{3})*(?:\s?[kK])?)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return "Not Specified"