import string
import spacy

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


