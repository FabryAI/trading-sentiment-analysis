import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Usa la directory globale di nltk, nessuna gestione manuale
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

def preprocess_dataframe(df):
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    return df
