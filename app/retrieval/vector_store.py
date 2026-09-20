import json
import os
from typing import List, Tuple

import faiss
import numpy as np

from app.retrieval.embeddings import embed_texts


VECTORSTORE_DIR = "data/vectorstore"


class FAISSVectorStore:

    def __init__(self, index_path: str, chunks_path: str):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.index = None
        self.chunks: List[str] = []

    def build(self, chunks: List[str]) -> None:
        if not chunks:
            raise ValueError("Cannot build vector store from empty chunks.")

        # Convert paper chunks into embeddings
        embeddings = embed_texts(chunks)

        vectors = np.asarray(
            embeddings,
            dtype="float32"
        )

        if vectors.ndim != 2:
            raise ValueError("Invalid embedding shape.")

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]

        # Inner Product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(vectors)

        self.chunks = chunks

        self.save()

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Vector index has not been built.")

        os.makedirs(
            os.path.dirname(self.index_path),
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            self.index_path
        )

        with open(
            self.chunks_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_path}"
            )

        self.index = faiss.read_index(
            self.index_path
        )

        with open(
            self.chunks_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.chunks = json.load(file)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:

        if self.index is None:
            self.load()

        if not self.chunks:
            return []

        # Convert the user question into an embedding
        query_embedding = embed_texts([query])

        query_vector = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_vector.ndim != 2:
            raise ValueError("Invalid query embedding shape.")

        faiss.normalize_L2(query_vector)

        k = min(
            top_k,
            len(self.chunks)
        )

        scores, indices = self.index.search(
            query_vector,
            k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):
            if index == -1:
                continue

            results.append(
                (
                    self.chunks[index],
                    float(score)
                )
            )

        return results


def create_vector_store(
    chunks: List[str],
    store_name: str = "paper_store"
) -> str:

    store_dir = os.path.join(
        VECTORSTORE_DIR,
        store_name
    )

    os.makedirs(
        store_dir,
        exist_ok=True
    )

    index_path = os.path.join(
        store_dir,
        "index.faiss"
    )

    chunks_path = os.path.join(
        store_dir,
        "chunks.json"
    )

    vector_store = FAISSVectorStore(
        index_path=index_path,
        chunks_path=chunks_path
    )

    vector_store.build(chunks)

    return store_dir


def search_vector_store(
    store_dir: str,
    query: str,
    top_k: int = 5
) -> List[Tuple[str, float]]:

    index_path = os.path.join(
        store_dir,
        "index.faiss"
    )

    chunks_path = os.path.join(
        store_dir,
        "chunks.json"
    )

    vector_store = FAISSVectorStore(
        index_path=index_path,
        chunks_path=chunks_path
    )

    return vector_store.search(
        query=query,
        top_k=top_k
    )