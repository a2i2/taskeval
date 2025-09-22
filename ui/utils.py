import re
from typing import Set, List, Dict

# -----------------------------
# HIGHLIGHT COLORS FOR EACH SECTION
# -----------------------------
HIGHLIGHT_COLORS = [
    "#fef08a",  # Yellow
    "#bfdbfe",  # Light Blue
    "#fed7d7",  # Light Pink
    "#bbf7d0",  # Light Green
    "#fed7aa",  # Light Orange
    "#e9d5ff",  # Light Violet
    "#fde68a",  # Light Amber
    "#fecaca",  # Light Red
]


# -----------------------------
# KEYWORD EXTRACTION AND HIGHLIGHTING FUNCTIONS
# -----------------------------
def extract_keywords_and_phrases(text: str, min_word_length: int = 3) -> Set[str]:
    """Extract meaningful keywords and phrases from text."""
    keywords = set()

    # Clean text and convert to lowercase
    cleaned_text = re.sub(r"[^\w\s]", " ", text.lower())
    # Replace any whitespace sequence (spaces, tabs, newlines) with single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    # Replace multiple spaces, tabs, newlines with single space
    cleaned_text = re.sub(r"[ \t\n\r\f\v]+", " ", cleaned_text)
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing spaces
    words = cleaned_text.split(" ")

    # Single words (filter out common words)
    stop_words = {
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "among",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "them",
        "their",
        "there",
        "where",
        "when",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "a",
        "an",
    }

    for word in words:
        if len(word) >= min_word_length and word not in stop_words:
            keywords.add(word)

    # Extract 2-word phrases
    for i in range(len(words) - 1):
        if len(words[i]) >= min_word_length and len(words[i + 1]) >= min_word_length:
            phrase = f"{words[i]} {words[i+1]}"
            if words[i] not in stop_words and words[i + 1] not in stop_words:
                keywords.add(phrase)

    # Extract 3-word phrases
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        if (
            words[i] not in stop_words
            and words[i + 1] not in stop_words
            and words[i + 2] not in stop_words
        ):
            keywords.add(phrase)

    # Extract numbers that are whole words (separated by word boundaries)
    numbers = re.findall(r"\b\d+\.?\d*%?\b", text)
    for num in numbers:
        keywords.add(num.lower())

    return keywords


def get_section_specific_matches(
    llm_output: str, extracted_sections: List[str]
) -> Dict[int, Set[str]]:
    """Get keywords that match between LLM output and each specific extracted section."""
    llm_keywords = extract_keywords_and_phrases(llm_output)

    section_matches = {}

    for i, section in enumerate(extracted_sections):
        section_keywords = extract_keywords_and_phrases(section)
        # Find matches for this specific section
        matches = set()
        for llm_kw in llm_keywords:
            for sect_kw in section_keywords:
                if llm_kw == sect_kw:
                    matches.add(llm_kw)

        if matches:
            section_matches[i] = matches

    return section_matches


def highlight_text_multicolor(text: str, section_matches: Dict[int, Set[str]]) -> str:
    """Highlight keywords in text with different colors for each section."""
    highlighted_text = text
    highlighted_text = re.sub(r"\s+", " ", highlighted_text)
    highlighted_text = highlighted_text.strip()

    # Create a list of all keywords with their associated colors
    keyword_color_map = {}
    for section_idx, keywords in section_matches.items():
        color = HIGHLIGHT_COLORS[section_idx]
        for keyword in keywords:
            if len(keyword) >= 3:  # Only meaningful keywords
                keyword_color_map[keyword] = color

    # Sort keywords by length (longest first) to avoid partial replacements
    sorted_keywords = sorted(keyword_color_map.keys(), key=len, reverse=True)

    for keyword in sorted_keywords:
        color = keyword_color_map[keyword]
        # Case-insensitive replacement
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
        highlighted_text = pattern.sub(
            f'<span style="background-color: {color}; color: #000; font-weight: 600; padding: 3px 6px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.2);">{keyword}</span>',
            highlighted_text,
        )

    return highlighted_text


def highlight_section_text(text: str, section_keywords: Set[str], color: str) -> str:
    """Highlight keywords in a specific section with its assigned color."""
    highlighted_text = text
    highlighted_text = re.sub(r"\s+", " ", highlighted_text)
    highlighted_text = highlighted_text.strip()

    # Sort keywords by length (longest first)
    sorted_keywords = sorted(section_keywords, key=len, reverse=True)

    for keyword in sorted_keywords:
        if len(keyword) >= 3:
            pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<span style="background-color: {color}; color: #000; font-weight: 600; padding: 3px 6px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.2);">{keyword}</span>',
                highlighted_text,
            )

    return highlighted_text
