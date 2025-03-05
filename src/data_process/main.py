import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from config import settings
import pandas as pd
from sqlalchemy import create_engine


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


def preprocess_text(text):
    text = text.lower()
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

if __name__ == "__main__":

    db_url = (
            f"postgresql://{settings.db_username}:{settings.db_password}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        )
    engine = create_engine(db_url)
    df = pd.read_sql_table('page_content', engine)
    text = df['extracted_content'].iloc[1]

    #preprocess the text
    processed_text = preprocess_text(text)
    print(text)
    print(processed_text)



