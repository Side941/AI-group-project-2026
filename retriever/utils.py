import json
import nltk # type: ignore
import string
from pathlib import Path

# Download necessary NLTK resources for tokenization
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load chunks from a JSON file
def load_chunks(json_path="knowledge_based/cddr_chunks.json"):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {json_path}")
    with open(path, 'r', encoding='utf-8') as f:
        try:
            chunks = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in knowledge base file: {e}")
    return chunks

# Tokenize text into lowercase words, removing punctuation
def tokenize(text):
    tokens = nltk.word_tokenize(text.lower())
    result = []
    for token in tokens:
        if token not in string.punctuation:
            result.append(token)
    return result