import string
import spacy
from typing import Optional

nlp = spacy.load('en_core_web_sm')

def remove_extra_spaces(text):
    return ' '.join(text.split())


def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_no_stops = [token.lemma_ for token in doc if not token.is_stop]
    # Remove punctuation tokens
    lemmatized_no_stops = [token for token in lemmatized_no_stops if token not in string.punctuation]
    # Remove numbers
    lemmatized_no_stops = [token for token in lemmatized_no_stops if not token.isdigit()]
    # Remove any special signs like @, #, $, %, etc.
    lemmatized_no_stops = [token for token in lemmatized_no_stops if not any(char in token for char in '@#$%&*()_+-=[]{}|;:,.<>?/')]
    return ' '.join(lemmatized_no_stops)


def _normalize_for_keywords(text: Optional[str]) -> str:
    """Normalize free text for keyword indexing/search.
    - Lowercase
    - Lemmatize and drop stopwords
    - Remove punctuation, digits, and special symbols
    - Collapse extra whitespace
    Returns empty string for None/empty inputs.
    """
    if not text:
        return ""
    lowered = text.lower()
    lemmatized = lemmatize_text(lowered)
    normalized = remove_extra_spaces(lemmatized)
    return normalized


# Backwards/forwards-compatible function names used across the codebase

def process_text_for_keyword_search(text: Optional[str]) -> str:
    return _normalize_for_keywords(text)


def process_text_for_keywords(text: Optional[str]) -> str:
    return _normalize_for_keywords(text)


