import os
from typing import List

import onnxruntime as ort
from fastembed import TextEmbedding


MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Reduce ONNX Runtime memory/thread pressure on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"


class EmbeddingModel:

    def __init__(self, model_name: str = MODEL_NAME):

        session_options = ort.SessionOptions()

        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        self.model = TextEmbedding(
            model_name=model_name,
            threads=1,
            session_options=session_options
        )

    def encode(self, texts: List[str]) -> List[List[float]]:

        if not texts:
            return []

        embeddings = self.model.embed(
            texts,
            batch_size=4
        )

        return [
            embedding.tolist()
            for embedding in embeddings
        ]


_embedding_model = None


def get_embedding_model() -> EmbeddingModel:

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = EmbeddingModel()

    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:

    model = get_embedding_model()

    return model.encode(texts)