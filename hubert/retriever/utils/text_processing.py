import re
import logging
import string
import spacy
nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)

def remove_extra_spaces(text):
    return ' '.join(text.split())

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_no_stops = [token.lemma_ for token in doc if not token.is_stop]
    # Remove punctuation tokens
    lemmatized_no_stops = [token for token in lemmatized_no_stops if token not in string.punctuation]
    return ' '.join(lemmatized_no_stops)

def process_text(text):
    text = remove_extra_spaces(text)
    text = lemmatize_text(text)
    text = text.lower()
    return text
