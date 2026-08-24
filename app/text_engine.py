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

    def clean_list_from_text(self, text: str, only_bullets: bool = True) -> list:
        """
        Converts a block of text into a clean list of strings.
        If only_bullets is True, ONLY returns lines that start with an explicit bullet or list marker.
        Strips bullet prefixes like "- ", "* ", "• ", "1. ", or "a)".
        """
        lines = text.split('\n')
        cleaned = []
        bullet_pattern = r'^\s*(?:[\-\*\•\▪\▸\–\—\>]|\d+[\.\)]|[a-zA-Z][\.\)])\s+'
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if only_bullets:
                if re.match(bullet_pattern, line):
                    clean_line = re.sub(bullet_pattern, '', line).strip()
                    if len(clean_line) > 3:
                        cleaned.append(clean_line)
            else:
                clean_line = re.sub(r'^[\s\-\*\•]+|^\d+[\.\)]\s*', '', line).strip()
                if len(clean_line) > 3:
                    cleaned.append(clean_line)
        return cleaned

    def detect_work_type(self, text: str) -> str:
        """Identifies Remote, Hybrid, or Onsite via keyword matching.
        ONLY returns Onsite if explicitly stated in text; otherwise returns Unknown.
        """
        if not text or not isinstance(text, str):
            return "Unknown"
        t = self._prepare_text(text)
        
        # Check for Remote explicitly
        if re.search(r'\bremote\b|\bwork[ -]from[ -]home\b|\bfreely[ -]remote\b|\bwork[ -]from[ -]anywhere\b|\bvirtual[ -]role\b|\bhome[ -]based\b|\bhome[ -]office\b|\bremote[ -]first\b|\bremote[ -]friendly\b|\blocation[ -]independent\b|\btelecommute\b', t):
            return "Remote"
        # Check for Hybrid explicitly
        if re.search(r'\bhybrid\b|\bdays[ -]in[ -]office\b|\bdays[ -]remote\b|\bcore[ -]days\b|\bad[ -]hoc\b|\bpartially[ -]remote\b', t):
            return "Hybrid"
        # Check for Onsite / In Office explicitly
        if re.search(r'\bonsite\b|\bon[ -]site\b|\bin[ -]office\b|\boffice[ -]based\b|\bin[ -]person\b', t):
            return "Onsite"
            
        return "Unknown"

    def detect_work_type_from_fields(self, *fields) -> str:
        """
        Checks multiple fields (e.g., flexibility, location, title, description) for explicit remote, hybrid, or onsite mentions.
        Prioritizes Remote/Hybrid if found in any field, and only returns Onsite if explicitly stated in one of the fields.
        """
        results = []
        for field in fields:
            if not field:
                continue
            if isinstance(field, (list, tuple)):
                field_str = " ".join(str(f) for f in field if f)
            else:
                field_str = str(field)
            res = self.detect_work_type(field_str)
            if res != "Unknown":
                results.append(res)

        if "Remote" in results:
            return "Remote"
        if "Hybrid" in results:
            return "Hybrid"
        if "Onsite" in results:
            return "Onsite"

        return "Unknown"


    def detect_seniority(self, text: str, title: str = "") -> str:
        """Identifies seniority level via keyword matching."""
        # 1. Check title first (highly reliable) using word boundaries
        if title:
            title_clean = self._prepare_text(title)
            title_rules = {
                "C-Suite": r'\bc-suite\b|\bexecutive\b|\bvp\b|\bvice[ -]president\b|\bchief\b',
                "Management": r'\bmanager\b|\bdirector\b|\bhead of\b|\blead\b',
                "Senior": r'\bsenior\b|\bsr\b|\bprincipal\b|\bstaff\b',
                "Mid-Level": r'\bintermediate\b|\bmid-level\b|\bmid\b|\bspecialist\b',
                "Junior": r'\bjunior\b|\bjr\b|\bentry level\b|\bassociate\b|\bintern\b'
            }
            for level, pattern in title_rules.items():
                if re.search(pattern, title_clean):
                    return level

        # 2. Check full text (description) using word boundaries to avoid body context fragments matching
        text_clean = self._prepare_text(text)
        rules = {
            "C-Suite": r'\bc-suite\b|\bexecutive\b|\bvp\b|\bvice[ -]president\b|\bchief officer\b',
            "Management": r'\bmanager\b|\bdirector\b|\bhead of\b|\blead\b',
            "Senior": r'\bsenior\b|\bsr\b|\bprincipal\b|\bstaff\b',
            "Mid-Level": r'\bintermediate\b|\bmid-level\b|\bspecialist\b',
            "Junior": r'\bjunior\b|\bjr\b|\bentry level\b|\bassociate\b|\bintern\b'
        }
        for level, pattern in rules.items():
            if re.search(pattern, text_clean):
                return level
        return "Unknown"

    def extract_salary(self, text: str) -> str:
        """Attempts to find a salary range in the text."""
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

        for tz, pattern in timezone_keywords.items():
            if re.search(pattern, text_lower):
                return tz

        utc_offset_match = re.search(r'(?:utc|gmt)\s?[+-]\d{1,2}(?::?(?:00|30))?', text_lower)
        if utc_offset_match:
            return utc_offset_match.group(0).upper()

        tz_phrases = [
            (r'must be (?:in|within|located in|based in) (?:the )?(?:us|usa|united states).*?(?:timezone|time|hours)', 'US Timezone'),
            (r'work (?:in|within) (?:the )?(?:eastern|central|mountain|pacific) (?:time|timezone)', None),
            (r'(?:eastern|central|mountain|pacific) (?:time|timezone)\s*(?:hours|preferred|required|standard)?', None),
        ]

        for phrase, fallback in tz_phrases:
            match = re.search(phrase, text_lower)
            if match:
                result = match.group(0)
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

        global_tz = re.search(r'work (?:from )?(?:anywhere|globally|worldwide)|(?:any|all) (?:timezone|time zone)', text_lower)
        if global_tz:
            return 'Any'

        return "Not Specified"

    def normalize_scraped_text(self, text: str) -> str:
        """
        Normalizes scraped job text where HTML tags were stripped without adding linebreaks.
        Inserts newlines before concatenated headers, colons, and section breaks.
        """
        if not text or not isinstance(text, str):
            return ""
        # 1. Split colons, periods, question marks, closing parens followed by uppercase
        t = re.sub(r'([:\?\!\.\)])\s*([A-Z0-9])', r'\1\n\2', text)
        # 2. Split known header phrase boundaries
        t = re.sub(r'([a-z0-9\.\?\)])(Requirements|Responsibilities|Qualifications|About|Who You Are|Key Responsibilities|Job Description|The Role|Position Overview|Essential Functions)', r'\1\n\2', t, flags=re.IGNORECASE)
        # 3. Split lowercase letter/digit followed by multi-word Title Case header (e.g. 'OpportunityFlywire is building')
        t = re.sub(r'([a-z0-9\.\?\)])([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}:?)', r'\1\n\2', t)
        return t

    def extract_section_bullets(self, text: str, header_patterns: list, stop_header_patterns: list = None) -> list:
        """
        Scans text for all header lines matching header_patterns.
        Matches partial phrase matches on short header lines.
        Collects explicit bullet items or line items under each header,
        stopping when reaching a major stop header or end of text.
        Combines and deduplicates bullets across all matching header sections.
        """
        if not text or not isinstance(text, str):
            return []

        text = self.normalize_scraped_text(text)

        combined_headers = "|".join(f"(?:{p})" for p in header_patterns)
        header_re = rf"(?im)^(?:[#\s\d\.\-]*).*?(?:{combined_headers})\b.*$"

        stop_re = None
        if stop_header_patterns:
            combined_stops = "|".join(f"(?:{p})" for p in stop_header_patterns)
            stop_re = rf"(?im)^(?:[#\s\d\.\-]*).*?(?:{combined_stops})\b.*$"

        matches = list(re.finditer(header_re, text))
        if not matches:
            return []

        all_bullets = []

        for match in matches:
            header_line = match.group(0).strip()
            # Ignore long body sentences
            if len(header_line) > 75 or (header_line.endswith('.') and not header_line.startswith('#')):
                continue

            start_idx = match.end()
            remainder = text[start_idx:]

            end_idx = len(remainder)
            if stop_re:
                for stop_match in re.finditer(stop_re, remainder):
                    stop_line = stop_match.group(0).strip()
                    # Skip bullet lines starting with list markers
                    if re.match(r'^\s*(?:[\-\*\•\▪\▸\–\—\>]|\d+[\.\)]|[a-zA-Z][\.\)])\s+', stop_line):
                        continue
                    if len(stop_line) <= 75 and not (stop_line.endswith('.') and not stop_line.startswith('#')):
                        end_idx = stop_match.start()
                        break

            section_text = remainder[:end_idx].strip()
            bullets = self.clean_list_from_text(section_text, only_bullets=True)
            if not bullets:
                # Fallback to short non-bulleted list items if no explicit bullet markers were found
                candidates = self.clean_list_from_text(section_text, only_bullets=False)
                bullets = [c for c in candidates if len(c) <= 350 and not c.endswith('.')]
            for b in bullets:
                if b not in all_bullets:
                    all_bullets.append(b)

        return all_bullets




    def extract_responsibilities(self, text: str) -> list:
        """Uses regex to extract responsibilities/duties bullets from text across all matching headers."""
        headers = [
            r'key responsibilities', r'responsibilities', r'duties & responsibilities',
            r'role & responsibilities', r'essential duties', r'what you\'ll do',
            r'what you will do', r"What you'll be doing", r'duties', r'job duties', 
            r'what you can expect to do', r'role overview', r'about the role',
            r'How you\'ll make an impact', r'Position Responsibilities', r'The Role',
            r'Essential Functions', r'Job Responsibilities', r'How will you make a difference',
            r'ACCOUNTABILITIES', r'IN THIS ROLE', r"What you'll work", r'Role Owns',
            r'Position Summary', r'Scope', r'POSITION OVERVIEW',
        ]
        stop_headers = [
            r'basic qualifications', r'minimum qualifications', r'preferred qualifications',
            r'required skills', r'qualifications', r'requirements', r'what we\'re looking for',
            r'what you\'ll need', r'what you bring', r'must have', r'required experience',
            r'skills & experience', r'skills and qualifications', r'desired skills',
            r'who you are', r'What makes you a good fit', r'Bonus points',
            r'Experience Requirements', r'Education Requirements', r'Highly Preferred Skills',
            r'What We\'re Looking For', r'Technical Experience', r'benefits', r'compensation',
            r'pay', r'salary', r'perks', r'about us', r'about the company', r'who we are',
            r'equal opportunity', r'how to apply'
        ]
        return self.extract_section_bullets(text, headers, stop_header_patterns=stop_headers)

    def extract_requirements(self, text: str) -> list:
        """Uses regex to extract requirements/qualifications bullets from text."""
        headers = [
            r'basic qualifications', r'minimum qualifications', r'preferred qualifications',
            r'required skills', r'qualifications', r'requirements', r'what we\'re looking for',
            r'what you\'ll need', r'what you bring', r'must have', r'required experience',
            r'skills & experience', r'skills and qualifications', r'desired skills',
            r'who you are',r'What makes you a good fit',r'Bonus points',r'qualifications',
            r'Experience Requirements',r'Education Requirements',r'Highly Preferred Skills',
            r"What We're Looking ForRequired", r'Technical Experience & Qualifications ',
            r"Experience you’ll need to have",r'Experience that would be great to have',
            r'Knowledge, Skills & Abilities',r'Essential/inimum qualifications',
            r'Essential experience required',r'valuable skills and experience',r'REQUIRED QUALIFICATIONS',
            r'Education and/or Experience',r'WE WILL EXPECT YOU TO HAVE',r'YOU MIGHT THRIVE IF',
            r'This might describe you',r'Nice to have',r"You’ll thrive if you",r'Requirements & Skills',
            r'education', r'experience',r'Must-haves',r'Nice-to-haves',r'What Sets the Right Candidate Apart',
            r'What we need to see',r'Ways to stand out from the crowd', r'Here\'s what we\'re looking for',
        ]
        return self.extract_section_bullets(text, headers)



    def extract_summary_from_description(self, text: str, title: str = "") -> str:
        """Extracts a concise deterministic summary from job description text."""
        if not text or not isinstance(text, str):
            return title or ""

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for p in paragraphs:
            if re.match(r'^(?:#+|\w+[\s:]*$)', p):
                continue
            clean_p = re.sub(r'\s+', ' ', p)
            if len(clean_p) > 30:
                if len(clean_p) > 300:
                    return clean_p[:297] + "..."
                return clean_p

        if title:
            return f"Position for {title}."
        return text[:200] if len(text) > 200 else text


