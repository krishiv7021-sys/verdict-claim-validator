import math
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


class FallbackTFIDFEmbedder:
    """
    High-reliability, deterministic TF-IDF vectorizer fallback.
    Used if sentence-transformers is downloading, offline, or unavailable.
    """
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit_transform(self, docs: List[str]) -> np.ndarray:
        # Build vocabulary
        doc_freq = {}
        tokenized_docs = []
        for doc in docs:
            tokens = set(self._tokenize(doc))
            tokenized_docs.append(self._tokenize(doc))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        n_docs = max(len(docs), 1)
        self.vocab = {t: idx for idx, t in enumerate(sorted(doc_freq.keys()))}
        self.idf = {t: math.log((n_docs + 1) / (count + 1)) + 1.0 for t, count in doc_freq.items()}

        vectors = []
        for tokens in tokenized_docs:
            vec = self._vectorize(tokens)
            vectors.append(vec)

        return np.array(vectors) if vectors else np.zeros((len(docs), 1))

    def transform(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            tokens = self._tokenize(text)
            vectors.append(self._vectorize(tokens))
        return np.array(vectors) if vectors else np.zeros((len(texts), max(len(self.vocab), 1)))

    def _tokenize(self, text: str) -> List[str]:
        import re
        words = re.findall(r'\b[a-zA-Z0-9_\-\$]+\b', text.lower())
        # Add character 3-grams for fuzzy matching
        ngrams = [text[i:i+3].lower() for i in range(len(text) - 2)]
        return words + ngrams[:50]

    def _vectorize(self, tokens: List[str]) -> np.ndarray:
        if not self.vocab:
            return np.zeros(1)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for t in tokens:
            if t in self.vocab:
                vec[self.vocab[t]] += self.idf.get(t, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec


class EmbeddingService:
    _instance = None

    def __new__(cls, model_name: str = "BAAI/bge-small-en-v1.5"):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.model = None
            cls._instance.fallback = FallbackTFIDFEmbedder()
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        if HAS_SENTENCE_TRANSFORMERS and SentenceTransformer is not None:
            try:
                # Try cached weights first to prevent offline stalls
                logger.info(f"Loading embedding model from cache: {self.model_name}")
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
                logger.info("Embedding model loaded from cache successfully.")
            except Exception:
                try:
                    logger.info(f"Loading embedding model: {self.model_name}")
                    self.model = SentenceTransformer(self.model_name)
                    logger.info("Embedding model loaded successfully.")
                except Exception as e:
                    logger.warning(f"Could not load SentenceTransformer ({e}). Using resilient TF-IDF fallback.")
                    self.model = None
        else:
            logger.info("sentence_transformers not installed. Using resilient TF-IDF fallback.")
            self.model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generates normalized vector embeddings for a list of strings.
        Returns 2D numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        if self.model is not None:
            try:
                # normalize_embeddings=True for cosine similarity
                embeddings = self.model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Model encode failed: {e}. Falling back to internal embedder.")

        # Resilient fallback
        if not self.fallback.vocab:
            return self.fallback.fit_transform(texts)
        return self.fallback.transform(texts)

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two 1D vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


# Module-level convenience functions
def get_embedding_service(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbeddingService:
    return EmbeddingService(model_name=model_name)
