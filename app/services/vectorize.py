from app.models.data import Data
from gensim.models import KeyedVectors
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")
model = KeyedVectors.load_word2vec_format("app/word2vec/GoogleNews-vectors-negative300.bin", binary=True)

GENRES_WEIGHT = 0.4
TAGS_WEIGHT = 0.4
DESC_WEIGHT = 0.2

VECTOR_SIZE = 300

def feature_vector(data: Data):
    try:
        genres = data.genres.split(' ')
        tags = data.tags.split(' ')
        description = data.description

        genres = [genre for genre in genres if genre in model]
        tags = [tag for tag in tags if tag in model]

        doc = nlp(description)
        res = []

        for token in doc:
            if token.is_stop or token.is_digit or token.is_punct:
                pass
            else:
                word = token.lemma_.lower()
                if word in model:
                    res.append(word)

        desc_vector = np.mean([model[word] for word in res], axis=0) if res else np.zeros(VECTOR_SIZE)
        genres_vector = np.mean([model[genre] for genre in genres], axis=0) if genres else np.zeros(VECTOR_SIZE)
        tags_vector = np.mean([model[tag] for tag in tags], axis=0) if tags else np.zeros(VECTOR_SIZE)
            
        result_vector = GENRES_WEIGHT * genres_vector + TAGS_WEIGHT * tags_vector + DESC_WEIGHT * desc_vector

        # Normalize for use with cosine similarity later
        norm = np.linalg.norm(result_vector)
        return result_vector / norm if norm > 0 else result_vector
    
    except Exception as e:
        print(f"Error vectorizing data: {e}")
        return np.zeros(VECTOR_SIZE)

