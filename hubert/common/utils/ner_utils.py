import spacy
from typing import Dict, List

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extracts named entities from a given text and groups them by their label.
    """
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        else:
            entities[ent.label_] = [ent.text]
    return entities

if __name__ == '__main__':
    # Example usage
    example_text = "Who is Stefan Lessmann and what is his role at the Humboldt University of Berlin?"
    extracted_entities = extract_entities(example_text)
    print(f"Extracted entities from: '{example_text}'")
    print(extracted_entities)

    example_text_2 = "Tell me about research papers on Large Language Models from 2023."
    extracted_entities_2 = extract_entities(example_text_2)
    print(f"Extracted entities from: '{example_text_2}'")
    print(extracted_entities_2) 