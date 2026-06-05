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
        """Extracts a block of text belonging to a specific header from markdown text."""
        pattern = rf"{re.escape(header_name)}[:\n\r]+([\s\S]*?)(?=\n[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*[:\n]|\Z)"
        match = re.search(pattern, text, re.IGNORECASE if not self.case_sensitive else 0)
        return match.group(1).strip() if match else ""

    def clean_list_from_text(self, text: str) -> list:
        """
        Converts a block of text into a clean list of strings. Helpful for vecotirzation
        This removes bullets like "- " or "* "
        """
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
        if re.search(r'\bremote\b|\bwork[ -]from[ -]home\b|\bfreely[ -]remote\b|\bwork[ -]from[ -]anywhere\b|\bvirtual[ -]role\b|\bhome[ -]based\b|\bhome[ -]office\b|\bremote[ -]first\b|\bremote[ -]friendly\b|\blocation[ -]independent', text):
            return "Remote"
        if re.search(r'\bhybrid\b|\bdays[ -]in[ -]office\b|\bdays[ -]remote\b|\bcore[ -]days\b|\bad[ -]hoc', text):
            return "Hybrid"
        if re.search(r'\bonsite\b|\bon[ -]site\b|\boffice[ -]based\b|\bin[ -]person', text):
            return "Onsite"
        return "Unknown"

    def detect_seniority(self, text: str) -> str:
        """Identifies seniority level via keyword matching."""
        text = self._prepare_text(text)
        # Order matters: check for higher levels first
        rules = {
            "C-Suite": r'c-suite|executive|vp|vice president|chief officer',
            "Management": r'manager|director|head of|lead',
            "Senior": r'senior|sr\.|sr |principal|staff',
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

    def detect_timezone(self, text: str) -> str:
        """
        Identifies timezone mentions in job descriptions.
        Looks for explicit timezone abbreviations, UTC offsets, and location
        references commonly used to indicate target timezone.
        Returns the detected timezone string or 'Not Specified'.
        """
        text_lower = text.lower()

        # 1. Check for explicit timezone abbreviations
        # Ordered roughly by prevalence in US job postings
        timezone_keywords = {
            "EST": r'\best\b',
            "EDT": r'\bedt\b',
            "ET":   r'\bet\b.*?(?:time|zone|hours)',
            "CST": r'\bcst\b',
            "CDT": r'\bcdt\b',
            "CT":   r'\bct\b.*?(?:time|zone|hours)',
            "MST": r'\bmst\b',
            "MDT": r'\bmdt\b',
            "MT":   r'\bmt\b.*?(?:time|zone|hours)',
            "PST": r'\bpst\b',
            "PDT": r'\bpdt\b',
            "PT":   r'\bpt\b.*?(?:time|zone|hours)',
            "GMT": r'\bgmt\b',
            "UTC": r'\butc\b',
            "CET": r'\bcet\b',
            "IST": r'\bist\b',
            "AEST": r'\baest\b',
            "AEDT": r'\baedt\b',
        }

        # Try explicit abbreviation matches first
        for tz, pattern in timezone_keywords.items():
            if re.search(pattern, text_lower):
                return tz

        # 2. Check for UTC offset patterns like "UTC-5", "UTC+1", "GMT-4"
        utc_offset_match = re.search(r'(?:utc|gmt)\s?[+-]\d{1,2}(?::?(?:00|30))?', text_lower)
        if utc_offset_match:
            return utc_offset_match.group(0).upper()

        # 3. Look for phrases that indicate the timezone requirement
        tz_phrases = [
            (r'must be (?:in|within|located in|based in) (?:the )?(?:us|usa|united states).*?(?:timezone|time|hours)', 'US Timezone'),
            (r'work (?:in|within) (?:the )?(?:eastern|central|mountain|pacific) (?:time|timezone)', None),
            (r'(?:eastern|central|mountain|pacific) (?:time|timezone)\s*(?:hours|preferred|required|standard)?', None),
        ]

        for phrase, fallback in tz_phrases:
            match = re.search(phrase, text_lower)
            if match:
                result = match.group(0)
                # Convert the matched phrase back into a clean timezone name
                if 'eastern' in result:
                    return 'ET'
                elif 'central' in result:
                    return 'CT'
                elif 'mountain' in result:
                    return 'MT'
                elif 'pacific' in result:
                    return 'PT'
                if fallback:
                    return fallback

        # 4. Check for global timezone phrases
        global_tz = re.search(r'work (?:from )?(?:anywhere|globally|worldwide)|(?:any|all) (?:timezone|time zone)', text_lower)
        if global_tz:
            return 'Any'

        return "Not Specified"
