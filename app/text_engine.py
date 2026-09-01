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

    def is_title_disqualified(self, title: str, disqualified_titles: list) -> bool:
        """Checks if a job title matches any disqualified word, phrase, or regex pattern."""
        from app.rule_filters import is_title_disqualified
        return is_title_disqualified(title, disqualified_titles)

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

    # Tech keywords preserved during normalizer word splitting
    TECH_PRESERVE = [
        'JavaScript', 'TypeScript', 'PostgreSQL', 'MongoDB', 'OpenAI', 'GraphQL',
        'Bitbucket', 'GitHub', 'GitLab', 'PowerBI', 'TensorRT', 'TensorFlow',
        'QuickBooks', 'Salesforce', 'DevOps', 'MLOps', 'FinOps', 'AutoML',
        'Netcracker', 'vLLM', 'PyTorch', 'Snowflake', 'Databricks', 'OpenRouter',
        'LangChain', 'FastAPI', 'Kubernetes', 'Microservices', 'Scikit-learn',
        'Webflow', 'DealCloud', 'Intapp', 'GFiber', 'Atlassian', 'CloudFormation',
        'Jenkins', 'SonarQube', 'Artifactory', 'Openshift', 'Docker', 'Redis',
        'Oracle', 'Ansible', 'Terraform', 'Pulumi', 'NextJS', 'NestJS', 'Next.js', 'Nest.js',
        'Claude', 'SageMaker', 'Kubeflow', 'Datadog', 'Prometheus', 'Grafana', 'ServiceNow',
        'Jira', 'Confluence', 'Smartsheet', 'Tableau', 'Power BI'
    ]

    BULLET_PREFIX_RE = r'^\s*(?:[\-\*\•\▪\▸\–\—\>]|\d+[\.\)]|[a-zA-Z][\.\)])\s+'

    METADATA_OR_HEADER_RE = re.compile(
        r'^(?:'
        r'Job Title|Location|Position Type|Salary Range|Experience Required|Sponsorship|'
        r'Application Information|Application deadline|Reporting to|Zone [A-Z]:?|'
        r'Job Band|Shift|Hours Per Week|Weekly Schedule|Referral Bonus Amount|'
        r'Minimum Education Requirement|EEO Statement|Equal Employment Opportunity|'
        r'Equal Opportunity|Powered by|Stay connected|Please note|About you|'
        r'Our Core Behaviors|Benefits|Benefits & Perks|Perks|Why Join Us|What Success Looks Like|'
        r'What you will gain.*|Here, you will have.*|Expand Your Skills|Enjoy Where You Work|'
        r'Support What Matters Most|How do you want to make your impact\?|'
        r'For jobs located in the United States.*|The following represents the expected range.*|'
        r'This role is eligible to participate.*|The estimated base salary.*|'
        r'The successful candidate’s starting salary.*|To comply with pay transparency.*|'
        r'TECHNICAL DEPTH & RISK MANAGEMENT|SNOWFLAKE-NATIVE TECH STACK|'
        r'OUR IDEAL CANDIDATE WILL HAVE|BONUS POINTS FOR HAVING.*|'
        r'What you\'ll be doing|What you will do|What you\'ll do|What you will be doing|'
        r'What you\'ll achieve|What you will achieve|'
        r'What you\'ll own|What you will own|'
        r'What we need to see|Ways to stand out from the crowd|'
        r'Required Qualifications|Desired Qualifications|Preferred Qualifications|'
        r'Required Skills|Core Engineering|AI / ML Expertise|Product Mindset|'
        r'AI-Enabled Development|Nice to Have|Nice-to-haves|Must-haves|Must-Have|Must have|'
        r'In this role, you will|In this role, you’ll|In this role, you\'ll|In this role you will|'
        r'Key Responsibilities|Responsibilities and Requirements|Responsibilities & Requirements|'
        r'Responsibilities|Essential Duties|Essential Functions|'
        r'Essential Job Functions|Essential Functions and Primary Duties|Essential Functions and Job Duties|'
        r'At a minimum, we’d like you to have|It’s preferred if you have|'
        r'ON DAY ONE WE WILL EXPECT YOU TO HAVE|WE WILL EXPECT YOU TO HAVE|'
        r'To apply for this.*|ABOUT OUR TEAM|About Our Team|Your background|'
        r'Frontend|Backend|Database|Cloud|Skills|Desirable Requirements|Essential Requirements|'
        r'What you bring|What you\'ll bring|What you will bring|'
        r'What we are looking for|What we\'re looking for|'
        r'Highly preferred|Bonus Points|Bonus if you have.*|'
        r'Experience, Education, Skills, and Competencies|Industry/Manufacturer Certifications|'
        r'Security Clearance Requirements|Clearance Requirements|'
        r'Travel Requirements|Physical Demands|Work Environment|Typical Projects|'
        r'What you can expect upon joining our team|Diversity Commitment|Our Pledge to Diversity|'
        r'The anticipated salary range|The anticipated base salary'
        r')[:\s]*$',
        re.IGNORECASE
    )

    BOILERPLATE_RE = re.compile(
        r'^(?:'
        r'As a [^,\n]+, you\'ll\.\.\.|'
        r'In this role, you will\.\.\.|'
        r'To apply for this [^,\n]+ role, you will ideally have:?|'
        r'How do you want to make your impact\?|'
        r'For jobs located in the United States, please visit.*|'
        r'The following represents the expected range.*|'
        r'This role is eligible to participate.*|'
        r'The estimated base salary range.*|'
        r'The successful candidate’s starting salary.*|'
        r'To comply with pay transparency.*|'
        r'Powered by JazzHR.*|'
        r'Apply today for immediate consideration.*|'
        r'Email your updated resume.*|'
        r'Call or Text.*|'
        r'Learn more about .*|'
        r'We look forward to connecting.*|'
        r'Stay connectedNot ready to apply.*|'
        r'Stay connected.*|'
        r'Please note:?|'
        r'We will ensure that individuals with disabilities.*|'
        r'To join [^,]+, you\'ll need a valid right to work.*|'
        r'If you are extended an offer.*|'
        r'For information about how [^,]+ processes.*|'
        r'0 -->|'
        r'H5Shift:.*|'
        r'1st shift \(United States of America\)|'
        r'Snowflake is growing fast, and we’re scaling our team.*|'
        r'We are looking for people who share our values.*|'
        r'Every Snowflake employee is expected to follow.*|'
        r'Snowflake employees must abide by the company’s data security plan.*|'
        r'It is every employee\'s duty to keep customer information.*|'
        r'Our Solution Engineers are customer obsessed.*|'
        r'We love to learn, are open to giving and receiving feedback.*|'
        r'Our team works to ensure data is accessible.*|'
        r'At GFiber, we believe that great internet.*|'
        r'GFiber is an Alphabet company.*|'
        r'The application window will be open until.*|'
        r'This opportunity will remain online based.*|'
        r'This role is not eligible for immigration sponsorship.*|'
        r'The US base salary range for this full-time position.*|'
        r'The US base.*|'
        r'As pay varies by location, your recruiter.*|'
        r'#LI-[A-Za-z0-9_\-]+.*|'
        r'GFiber is committed to equal opportunity employment.*|'
        r'Disclosure is voluntary, and this information.*|'
        r'For more information please refer to our Equal Employment Opportunity.*|'
        r'It\'s important to us to create an accessible.*|'
        r'If you have a need that requires accommodation.*|'
        r'Our candidate accommodations team will then connect.*|'
        r'Your base salary will be determined based on your location.*|'
        r'The base salary range is .*|'
        r'You will also be eligible for equity andbenefits.*|'
        r'Applications for this job will be accepted at least until.*|'
        r'This posting is for an existing vacancy.*|'
        r'NVIDIA uses AI tools in its recruiting processes.*|'
        r'NVIDIA is committed to fostering an inclusive work environment.*|'
        r'As we highly value diversity in our current.*|'
        r'Join us at NVIDIA, where we are defining the next era.*|'
        r'The salary for this position is expected to range.*|'
        r'The final salary offered is determined based on factors.*|'
        r'Novartis may change the published.*|'
        r'Your compensation will include a performance-based cash incentive.*|'
        r'US-based eligible employees will receive a comprehensive benefits.*|'
        r'In addition, employees are eligible for a generous time off package.*|'
        r'To learn more about the culture, rewards and benefits.*|'
        r'The Novartis Group of Companies are committed to working with.*|'
        r'If, because of a medical condition or disability, you need.*|'
        r'Please include the job requisition number in your message.*|'
        r'Salary Range\$[0-9,\.\s\-]+|'
        r'Skills DesiredCross-Functional Collaboration.*|'
        r'Remote-first \(United States\)|'
        r'Full-time PermanentExempt.*|'
        r'United States  \(all figures cited below.*|'
        r'Zone [A-Z]:.*|'
        r'Intapp is looking for intellectually curious.*|'
        r'(?:This\s+)?job description is not (?:intended|designed) to be all inclusive.*|'
        r'Please note this job description is not designed to cover.*|'
        r'Duties, responsibilities, and activities may change.*|'
        r'At any time, employees may perform other related duties.*|'
        r'.*does not accept unsolicited resumes.*|'
        r'Any unsolicited resume submitted.*|'
        r'Medical or recreational marijuana use is considered illegal.*|'
        r'See article.*|'
        r'Disclaimer:.*|'
        r'Consenting to our Applicant Data Privacy Policy.*|'
        r'By clicking (?:“|\")Submit Application(?:”|\").*'
        r')$',
        re.IGNORECASE
    )

    COMMON_STOP_HEADERS = [
        r'benefits', r'perks', r'compensation', r'salary', r'pay range', r'base pay',
        r'why join us', r'what you will gain', r'what we offer', r'what success looks like',
        r'about us', r'about the company', r'about our team', r'about you', r'who we are',
        r'our culture', r'our core behaviors', r'our values', r'working at',
        r'equal opportunity', r'eeo statement', r'equal employment opportunity',
        r'accessibility and reasonable accommodations', r'accommodations',
        r'how to apply', r'apply today', r'application information', r'application deadline',
        r'please note', r'stay connected', r'powered by', r'confidentiality',
        r'how do you want to make your impact', r'to comply with pay transparency',
        r'job band', r'referral bonus', r'minimum education requirement',
        r'your base salary', r'the salary for this position', r'salary range',
        r'the us base salary', r'nvidia is committed', r'snowflake is growing fast',
        r'about our team', r'about you', r'our core behaviors',
        r'travel requirements', r'physical demands', r'work environment',
        r'typical projects', r'what you can expect upon joining our team',
        r'diversity commitment', r'our pledge to diversity',
        r'applicant data privacy statement', r'note:', r'disclaimer:',
        r'notion is committed', r'a note on ai',
        r'the annual full time base', r'the anticipated (?:base )?salary',
        r'note on on-call'
    ]

    RESPONSIBILITY_HEADERS = [
        r'key responsibilities', r'responsibilities and requirements', r'responsibilities & requirements',
        r'responsibilities', r'duties & responsibilities',
        r'role & responsibilities', r'essential duties', r'what you\'ll do',
        r'what you will do', r"what you'll be doing", r"what you will be doing",
        r'what you\'ll achieve', r'what you will achieve',
        r'what you\'ll own', r'what you will own',
        r'these are the types of things you[\'’]ll be working on',
        r'duties', r'job duties', r'what you can expect to do', r'role overview',
        r'how you\'ll make an impact', r'position responsibilities',
        r'essential (?:job\s+)?functions(?:\s+and\s+(?:primary|job)\s+duties)?',
        r'essential (?:job\s+)?duties',
        r'essential functions', r'job responsibilities',
        r'how will you make a difference', r'accountabilities',
        r'in this role you will get to', r'in this role, you will', r'in this role, you\'ll', r'in this role, you’ll',
        r"what you'll work", r'role owns',
        r'as an? [^\n]+?, you(?:[\'’]ll|\.\.\.| can expect to)',
        r'as [^,\n]+?, your mission (?:will be|is) to:?'
    ]

    REQUIREMENT_HEADERS = [
        r'basic qualifications', r'minimum qualifications', r'preferred qualifications',
        r'required skills', r'qualifications', r'requirements',
        r'what we\'re looking for', r'what we are looking for',
        r'what you\'ll need', r'what you will need',
        r'skills you(?:[\'’]ll| will)? need(?: to bring)?',
        r'what you bring', r'what you\'ll bring', r'what you will bring',
        r'must have', r'must-have', r'required experience',
        r'skills & experience', r'skills and qualifications', r'desired skills',
        r'who you are', r'what makes you a good fit', r'bonus points', r'bonus points for having',
        r'experience requirements', r'education requirements', r'highly preferred(?: skills)?',
        r'highly preferred',
        r'this role is a (?:perfect|great|good) match for you if[^\n:]*',
        r'bonus if you have[^\n:]*',
        r'experience, education, skills, and competencies',
        r'industry/manufacturer certifications',
        r'certifications', r'competencies',
        r'security clearance requirements', r'clearance requirements',
        r"what we're looking forrequired", r'technical experience & qualifications',
        r"experience you’ll need to have", r"experience you\'ll need to have",
        r'experience that would be great to have', r'knowledge, skills & abilities',
        r'essential/minimum qualifications', r'essential experience required',
        r'valuable skills and experience', r'required qualifications', r'desired qualifications',
        r'essential requirements', r'desirable requirements',
        r'education and/or experience', r'we will expect you to have',
        r'on day one we will expect you to have', r'you might thrive if',
        r'this might describe you', r'nice to have', r'nice-to-haves', r'nice-to-have', r'must-haves',
        r"you’ll thrive if you", r"you'll thrive if you",
        r'what sets the right candidate apart', r'what we need to see',
        r'ways to stand out from the crowd', r'ways to stand out',
        r'here\'s what we\'re looking for', r'our ideal candidate will have',
        r'at a minimum, we’d like you to have', r'at a minimum, we\'d like you to have',
        r'it’s preferred if you have', r'it\'s preferred if you have',
        r'to apply for this [^:\n]+ role, you will ideally have',
        r'your background', r'skills desired', r'skills:'
    ]

    def normalize_scraped_text(self, text: str) -> str:
        """
        Normalizes scraped job text where HTML tags were stripped without adding linebreaks.
        Inserts newlines before concatenated headers, colons, bullet start action verbs, and section breaks.
        """
        if not text or not isinstance(text, str):
            return ""
        
        t = text
        # 1. Clean HTML entities, non-breaking spaces, zero-width chars, carriage returns
        t = t.replace('\u200b', ' ').replace('\xa0', ' ').replace('\r\n', '\n').replace('\r', '\n')
        # Pad 4-byte unicode emojis with spaces
        t = re.sub(r'([\U00010000-\U0010ffff])', r' \1 ', t)
        
        # Clean CSS selectors and rules
        t = re.sub(r'([A-Za-z]+)(body|html|style|table)\s*\{', r'\1 \2 {', t, flags=re.IGNORECASE)
        t = re.sub(r'(?:\b(?:body|html|strong|em|ul|ol|li|p|div|span|table|td|tr|th)\b|\.[a-z0-9_\-]+|\#[a-z0-9_\-]+)(?:\s*,\s*(?:\b(?:body|html|strong|em|ul|ol|li|p|div|span|table|td|tr|th)\b|\.[a-z0-9_\-]+|\#[a-z0-9_\-]+))*\s*\{[^}]*\}', ' ', t, flags=re.IGNORECASE)

        # Clean #LI- tags (handle jammed words like #LI-RemoteNote: -> #LI-Remote\nNote:)
        t = re.sub(r'(#LI-[A-Za-z0-9_\-]+?)([A-Z][a-z]+:?)', r'\1\n\2', t)
        t = re.sub(r'#LI-[A-Za-z0-9_\-]+', '', t)

        # Protect tech terms with exact casing before any word-splitting regex
        tech_map = {}
        def _protect(m):
            placeholder = f"__TECH_TERM_{len(tech_map)}__"
            tech_map[placeholder] = m.group(0)
            return placeholder
            
        for term in self.TECH_PRESERVE:
            t = re.sub(rf'\b{re.escape(term)}\b', _protect, t, flags=re.IGNORECASE)
        
        # 2. Add newlines after punctuation followed by capital letters, digits, bullets, or emojis
        t = re.sub(r'([:\?\!\.\)])\s*([A-Z0-9\-\*\•\▪\▸\–\—]|[\U00010000-\U0010ffff])', r'\1\n\2', t)
        
        # 3. Known headers and section demarcation phrases
        known_headers = [
            r'What you\'ll be doing', r'What you will be doing', r'What you\'ll do', r'What you will do',
            r'What you\'ll achieve', r'What you will achieve',
            r'What you\'ll own', r'What you will own',
            r'What we need to see', r'What we\'re looking for', r'What we are looking for',
            r'What you bring', r'What you\'ll bring', r'What you will bring',
            r'What you\'ll need', r'What you will need',
            r'Skills you\'ll need to bring', r'Skills you will need to bring', r'Skills you\'ll need', r'Skills you will need',
            r'Ways to stand out from the crowd', r'Ways to stand out',
            r'ON DAY ONE WE WILL EXPECT YOU TO HAVE', r'WE WILL EXPECT YOU TO HAVE',
            r'IN THIS ROLE YOU WILL GET TO', r'In this role, you will', r'In this role, you\'ll', r'In this role, you’ll', r'In this role you will',
            r'These are the types of things you\'ll be working on', r'These are the types of things you’ll be working on',
            r'As a [^,\n]+, you\'ll\.\.\.', r'As an? [^\n]+?, you(?:[\'’]ll|\.\.\.|:)',
            r'As [^,\n]+?, your mission (?:will be|is) to:?',
            r'This role is a (?:perfect|great|good) match for you if[^\n:]*',
            r'Key Responsibilities', r'Responsibilities and Requirements', r'Responsibilities & Requirements',
            r'Responsibilities', r'Role & Responsibilities', r'Essential Duties',
            r'Essential Job Functions', r'Essential Functions and Primary Duties', r'Essential Functions and Job Duties',
            r'Essential Functions', r'Job Responsibilities', r'Position Responsibilities',
            r'Required Qualifications', r'Desired Qualifications', r'Preferred Qualifications',
            r'Basic Qualifications', r'Minimum Qualifications', r'Required Skills', r'Required Experience',
            r'Requirements & Skills', r'Requirements', r'Qualifications', r'Essential Requirements',
            r'Desirable Requirements', r'Must-haves?', r'Must have', r'Must-Have', r'Nice-to-haves?', r'Nice to have', r'Nice-to-Have',
            r'Highly preferred', r'Bonus if you have[^\n:]*', r'Bonus Points', r'Bonus points for having', r'Bonus points',
            r'Our Ideal Candidate Will Have',
            r'At a minimum, we’d like you to have', r'At a minimum, we\'d like you to have',
            r'It’s preferred if you have', r'It\'s preferred if you have',
            r'To apply for this [^:\n]+ role, you will ideally have',
            r'Experience, Education, Skills, and Competencies', r'Industry/Manufacturer Certifications',
            r'Certifications', r'Competencies',
            r'Security Clearance Requirements', r'Clearance Requirements',
            r'Travel Requirements', r'Physical Demands', r'Work Environment',
            r'Typical Projects', r'Other Duties as Assigned', r'Other Duties',
            r'System Engineering \(\d+%\)', r'Technical Support \(\d+%\)', r'Sales Consultation \(\d+%\)',
            r'Cloud-to-on-prem migration strategy', r'Pre sales support and post sales execution', r'Training curriculum design & delivery',
            r'Your background', r'Who you are', r'About you', r'About our team', r'ABOUT OUR TEAM', r'About us',
            r'About the company', r'About the role', r'Why Join Us', r'What Success Looks Like',
            r'What you will gain at [^:\n]+', r'What you will gain', r'Benefits & Perks', r'Benefits', r'Perks', r'Compensation',
            r'The US base salary range', r'The base salary range', r'Salary Range',
            r'EEO Statement', r'Equal Employment Opportunity', r'Equal Opportunity',
            r'Accessibility and reasonable accommodations', r'Please note', r'Stay connected',
            r'How do you want to make your impact\?', r'How you\'ll make an impact',
            r'TECHNICAL DEPTH & RISK MANAGEMENT', r'SNOWFLAKE-NATIVE TECH STACK',
            r'Job Description Summary', r'Job Description', r'Job Summary', r'Position Overview', r'Role Description',
            r'Skills Desired', r'Skills:', r'Minimum Education Requirement', r'Job Band',
            r'Hours Per Week', r'Weekly Schedule', r'Referral Bonus Amount', r'Position Type',
            r'Salary Range:', r'Location:', r'Experience Required:', r'Sponsorship:',
            r'Application Information', r'Application deadline', r'Note:', r'Disclaimer:',
            r'Diversity Commitment', r'Our Pledge to Diversity',
            r'What you can expect upon joining our team',
            r'The anticipated salary range', r'The anticipated base salary',
            r'Disclaimer', r'Note on on-call'
        ]
        
        combined_known = "|".join(f"(?:{h})" for h in sorted(known_headers, key=len, reverse=True))

        # Major section headers that can start a section even with a space before them
        major_headers = [
            r'What you\'ll be doing', r'What you will be doing', r'What you\'ll do', r'What you will do',
            r'What you\'ll achieve', r'What you will achieve', r'What you\'ll own', r'What you will own',
            r'What we need to see', r'What we\'re looking for', r'What we are looking for',
            r'What you bring', r'What you\'ll bring', r'What you will bring',
            r'What you\'ll need', r'What you will need',
            r'Skills you\'ll need to bring', r'Skills you will need to bring',
            r'Key Responsibilities', r'Responsibilities and Requirements',
            r'Diversity Commitment', r'Our Pledge to Diversity',
            r'What you can expect upon joining our team',
            r'Must have', r'Must-have', r'Nice to have', r'Nice-to-have'
        ]
        combined_major = "|".join(f"(?:{h})" for h in sorted(major_headers, key=len, reverse=True))

        # Punctuation followed by header
        t = re.sub(rf'([:\?\!\.\)])\s*({combined_known})(?:\b|(?=[A-Z0-9\-\*\•\▪\▸\–\—]))', r'\1\n\2', t, flags=re.IGNORECASE)
        # Lowercase/digit jammed directly against header without space
        t = re.sub(rf'([a-z0-9])({combined_known})(?:\b|(?=[A-Z0-9\-\*\•\▪\▸\–\—]))', r'\1\n\2', t, flags=re.IGNORECASE)
        # Lowercase/digit/emoji followed by spaces and a major header phrase
        t = re.sub(rf'([a-z0-9]|[\U00010000-\U0010ffff])[ \t]+({combined_major})(?:\b|(?=[A-Z0-9\-\*\•\▪\▸\–\—]))', r'\1\n\2', t, flags=re.IGNORECASE)
        # Ellipsis followed by capital letter
        t = re.sub(r'(\.\.\.)\s*([A-Z])', r'\1\n\2', t)

        # Header followed on same line by content (capital letter, digit, bullet marker, or emoji) -> newline after header
        t = re.sub(rf'((?i:\b(?:{combined_known})))[ \t]*(?::[ \t]*)?((?:-(?:[ \t]+|[A-Z0-9])|[\*\•\▪\▸\–\—]|[\U00010000-\U0010ffff]|[A-Z0-9]))', r'\1:\n\2', t)
        
        # 5. Split known bullet action verbs and phrases when glued to lowercase/digits
        action_verbs = [
            r'Builds?', r'Designs?', r'Manages?', r'Leads?', r'Partners?', r'Drives?', r'Develops?',
            r'Supports?', r'Implements?', r'Collaborates?', r'Provides?', r'Serves?', r'Architects?',
            r'Evaluates?', r'Coordinates?', r'Escalates?', r'Monitors?', r'Produces?', r'Works?',
            r'Leverages?', r'Identifies?', r'Experience', r'Proficiency', r'Strong', r'Proven',
            r'Bachelor', r'Master', r'Ability', r'Familiarity', r'Knowledge', r'Working',
            r'Excellent', r'Demonstrated', r'Outstanding', r'Possesses', r'Broad range',
            r'Hands-on', r'University degree', r'Masters in', r'Expect approximately',
            r'Present', r'Work hands-on', r'Immerse', r'Take ownership', r'Documents?',
            r'participate in customer discovery', r'investigate, discover', r'have a broad understanding',
            r'Probe for', r'Be a product', r'Lead compelling', r'Understand, lead',
            r'Proactively forge', r'Help lead', r'Understand, track', r'Continuously learn',
            r'Apply agentic', r'Occasional work', r'Minimum of a', r'At least \d+',
            r'Solid understanding', r'Containerized solutions', r'Professional-level',
            r'Proficient in', r'Background with', r'Clustering or',
            r'Solid knowledge', r'Advance(?:d)? knowledge', r'Competencies typically',
            r'Has a value-driven', r'Presentation skills', r'Proficient use',
            r'Must be able', r'High School diploma', r'\d+\+? years', r'\d+ to \d+ years',
            r'Conduct outbound', r'Qualify and progress', r'Find leverage', r'Automate the parts',
            r'Identify and test', r'Refine how we', r'Sequence design', r'Ensure HubSpot',
            r'Convert event', r'Track record', r'Comfort with modern', r'Enterprise or multi',
            r'Excellent written', r'Are sharp', r'Learn something', r'Are genuinely curious',
            r'Are comfortable', r'Confident telephone', r'Have automated', r'Are persistent',
            r'Would rather sharpen', r'Care about the mission', r'Define the strategy',
            r'Identify AWS', r'Determine the scope', r'Translate the ground',
            r'Advise the customer', r'Act as the primary', r'Act as primary',
            r'Design a training', r'Develop training', r'Conduct in-person',
            r'Deep experience', r'Strong systems', r'Comfortable operating',
            r'Willingness and ability', r'Spanish language', r'Create detailed',
            r'Utilize D-tools', r'Prepare custom', r'Complete formal', r'Travel to client',
            r'Responsible for the overall', r'Provide technical', r'On-site service',
            r'Answer questions', r'Attend customer', r'Advise customers',
            r'In depth knowledge', r'Desire to grow', r'Ability to organize',
            r'D-Tools experience', r'CTS-D', r'QSC Level', r'Dante Level', r'Biamp Tesiraforte',
            r'Go- Primary', r'React / TypeScript-', r'Accessible interface', r'Automed Testing',
            r'PostgreSQL- schema', r'AWS -Or', r'Terraform - Infrastructure',
            r'CI/CD pipeline', r'Redis or comparable', r'OpenTelemetry instrumentation',
            r'Progressive delivery', r'Experience responding', r'Experience delivering',
            r'API gateway', r'Screen reader', r'US Citizenship', r'US Residency',
            r'Able to meet'
        ]
        combined_actions = "|".join(action_verbs)
        t = re.sub(rf'([a-z0-9\.\?\!\:\)])\s*({combined_actions})\b', r'\1\n\2', t)

        # 6. Split numbered list items
        t = re.sub(r'([a-zA-Z0-9\.\)])\s*(\d+\.\s+[A-Z])', r'\1\n\2', t)

        # 7. Split sub-headers like "Frontend:React, Next.jsBackend:Node.js"
        t = re.sub(r'([a-zA-Z0-9\.\)])\s*(Frontend:|Backend:|Database:|Cloud:|Skills:|Distributed Systems:|Predictive Failure Analysis:|Governance:)', r'\1\n\2', t, flags=re.IGNORECASE)
        
        # 8. Spacing for jammed words within sentences
        t = re.sub(r'([a-z]{2,})([A-Z][a-z]{2,})', r'\1 \2', t)
        
        # 9. Restore protected tech terms
        for placeholder, original in tech_map.items():
            t = t.replace(placeholder, original)
            
        return t

    def _clean_and_filter_item(self, line: str) -> str:
        """Helper to sanitize and filter extracted list items and bullets."""
        if not line or not isinstance(line, str):
            return ""
        if line.startswith('#'):
            return ""
        c = re.sub(self.BULLET_PREFIX_RE, '', line).strip()
        c = c.strip('\'"` ')
        c = re.sub(r'\s*Disclaimer:?\s*$', '', c, flags=re.IGNORECASE).strip()
        c = re.sub(r'\s*Note:?\s*$', '', c, flags=re.IGNORECASE).strip()
        if not re.search(r'[A-Za-z0-9]', c):
            return ""
        if len(c) < 4:
            return ""
        if self.METADATA_OR_HEADER_RE.match(c):
            return ""
        if self.BOILERPLATE_RE.match(c):
            return ""
        if c.endswith(':') and len(c) < 50:
            return ""
        if re.match(r'^(?:Job Band|Shift|Hours Per Week|Weekly Schedule|Referral Bonus Amount|Minimum Education Requirement|Location|Position Type|Salary Range|Sponsorship|Application deadline|Application Information|Reporting to):', c, re.IGNORECASE):
            return ""
        return c

    def extract_section_bullets(self, text: str, header_patterns: list, stop_header_patterns: list = None, is_requirement_extraction: bool = False) -> list:
        """
        Scans text for all header lines matching header_patterns.
        Collects explicit bullet items or line items under each header,
        stopping when reaching a major stop header or end of text.
        Combines and deduplicates bullets across all matching header sections.
        """
        if not text or not isinstance(text, str):
            return []

        norm_text = self.normalize_scraped_text(text)

        combined_headers = "|".join(f"(?:{p})" for p in header_patterns)
        header_re = rf"(?im)^(?:[#\s\d\.\-]*).*?(?:{combined_headers})\b.*$"

        stop_re = None
        if stop_header_patterns:
            combined_stops = "|".join(f"(?:{p})" for p in stop_header_patterns)
            stop_re = rf"(?im)^(?:[#\s\d\.\-]*).*?(?:{combined_stops})\b.*$"

        matches = list(re.finditer(header_re, norm_text))
        if not matches:
            return []

        all_bullets = []

        for match in matches:
            header_line = match.group(0).strip()
            # Ignore overly long sentences containing a keyword
            if len(header_line) > 85 and not header_line.startswith('#'):
                continue

            # If extracting requirements, exclude headers that are primarily responsibilities
            if is_requirement_extraction:
                if re.search(r'\bresponsibilit(?:y|ies)\b', header_line, re.I) and not re.search(r'\bqualifications\b', header_line, re.I):
                    continue
                if re.search(r'\btravel\s+requirements\b', header_line, re.I):
                    continue

            start_idx = match.end()
            remainder = norm_text[start_idx:]

            end_idx = len(remainder)
            if stop_re:
                for stop_match in re.finditer(stop_re, remainder):
                    stop_line = stop_match.group(0).strip()
                    starts_with_stop = any(re.match(rf'^(?:[#\s\d\.\-]*)(?:{p})(?:\b|:|\s|$)', stop_line, re.I) for p in (stop_header_patterns or []))
                    if starts_with_stop:
                        end_idx = stop_match.start()
                        break
                    if len(stop_line) > 85 and not stop_line.startswith('#'):
                        continue
                    if re.match(self.BULLET_PREFIX_RE, stop_line):
                        continue
                    end_idx = stop_match.start()
                    break

            section_text = remainder[:end_idx].strip()
            lines = section_text.split('\n')

            # Check if section text contains explicit bullet markers
            has_explicit_bullets = any(re.match(self.BULLET_PREFIX_RE, l.strip()) for l in lines if l.strip())
            
            section_items = []
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                
                # Check markdown headers
                if line_str.startswith('#'):
                    if any(re.search(rf'\b{p}\b', line_str, re.I) for p in header_patterns):
                        continue
                    break
                
                if has_explicit_bullets:
                    if re.match(self.BULLET_PREFIX_RE, line_str):
                        cleaned = self._clean_and_filter_item(line_str)
                        if cleaned and cleaned not in section_items:
                            section_items.append(cleaned)
                else:
                    # Non-explicit bullet items
                    cleaned = self._clean_and_filter_item(line_str)
                    if cleaned and cleaned not in section_items:
                        section_items.append(cleaned)

            # If no explicit bullets exist and section is a single unstructured paragraph, reject (return [])
            if not has_explicit_bullets:
                # If all section items look like a single continuous prose paragraph with <= 2 lines ending in period without list structures
                if len(section_items) <= 2 and all(item.endswith('.') for item in section_items) and not any(re.match(r'^(?:[0-9]+[\.\)]|[A-Z][a-z]+ing|\b(?:Build|Design|Manage|Lead|Partner|Drive|Develop|Support|Implement|Collaborate|Provide|Serve|Architect|Evaluate|Coordinate|Escalate|Monitor|Produce|Experience|Proficiency|Strong|Proven|Bachelor|Master|Ability|Familiarity|Knowledge|Working|Excellent|Demonstrated)\b)', item) for item in section_items):
                    section_items = []

            for item in section_items:
                if item not in all_bullets:
                    all_bullets.append(item)

        return all_bullets

    def extract_responsibilities(self, text: str) -> list:
        """Uses regex to extract responsibilities/duties bullets from text across all matching headers."""
        stop_headers = self.REQUIREMENT_HEADERS + self.COMMON_STOP_HEADERS
        return self.extract_section_bullets(text, self.RESPONSIBILITY_HEADERS, stop_header_patterns=stop_headers, is_requirement_extraction=False)

    def extract_requirements(self, text: str) -> list:
        """Uses regex to extract requirements/qualifications bullets from text."""
        stop_headers = self.RESPONSIBILITY_HEADERS + self.COMMON_STOP_HEADERS
        return self.extract_section_bullets(text, self.REQUIREMENT_HEADERS, stop_header_patterns=stop_headers, is_requirement_extraction=True)



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


