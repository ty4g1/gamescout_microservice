from app.models.data import Data
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

GENRES_WEIGHT = 0.4
TAGS_WEIGHT = 0.4
DESC_WEIGHT = 0.2

VECTOR_SIZE = 384

def feature_vector(data: Data):
    try:
        genres_text = ' '.join(data.genres.split(' ')) if data.genres else ""
        tags_text = ' '.join(data.tags.split(' ')) if data.tags else ""
        description = data.description if data.description else ""
        
        # Get embeddings (384d)
        genres_vector = model.encode(genres_text) if genres_text else np.zeros(VECTOR_SIZE)
        tags_vector = model.encode(tags_text) if tags_text else np.zeros(VECTOR_SIZE)
        desc_vector = model.encode(description) if description else np.zeros(VECTOR_SIZE)
        
        result_vector = (GENRES_WEIGHT * genres_vector + 
                        TAGS_WEIGHT * tags_vector + 
                        DESC_WEIGHT * desc_vector)

        # Normalize for use with cosine similarity later
        norm = np.linalg.norm(result_vector)
        return result_vector / norm if norm > 0 else result_vector
    
    except Exception as e:
        print(f"Error vectorizing data: {e}")
        return np.zeros(VECTOR_SIZE)

