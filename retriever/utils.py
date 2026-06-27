import json
import nltk # type: ignore
import string
from pathlib import Path

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def load_chunks(json_path="knowledge_based/icd11_chunks.json"):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {json_path}")
    with open(path, 'r', encoding='utf-8') as f:
        try:
            chunks = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in knowledge base file: {e}")

    for chunk in chunks:
        chunk['id'] = f"{chunk.get('disorder_code', 'unknown')}_{chunk.get('section', 'unknown').lower().replace(' ', '_')}"
        # Keep original short text for prompt injection
        chunk['prompt_text'] = chunk.get('text', '')
        # Use richer embed_text for retrieval indexing
        chunk['text'] = chunk.get('embed_text') or chunk.get('prompt_text', '')

    return chunks

def tokenize(text):
    tokens = nltk.word_tokenize(text.lower())
    result = []
    for token in tokens:
        if token not in string.punctuation:
            result.append(token)
    return result