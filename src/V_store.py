
import json
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:

    def __init__(self, dimension):
        """
        Initialize an empty FAISS index.

        dimension: number of dimensions in each embedding.
        """
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add_documents(self, chunks, embeddings):
        """
        Add document embeddings and their corresponding text chunks.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.index.d:
            raise ValueError("Embedding dimension does not match index.")

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings contain invalid values.")

        if len(embeddings) == 0:
            return

        # normalized
        norms = np.linalg.norm(embeddings, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Embeddings must be normalized.")

        embeddings = np.ascontiguousarray(embeddings) #encure correct memory layout, a memo layout operation 
        self.index.add(embeddings)

        self.chunks.extend(chunks)

    def save(self, directory="data/V_store"):
        """
        Save the FAISS index and corresponding text chunks.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss")
        )

        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, directory="data/V_store"):
        """
        Load an existing FAISS index and its text chunks.
        """
        directory = Path(directory)

        index = faiss.read_index(
            str(directory / "index.faiss")
        )

        with open(directory / "chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

        store = cls(index.d)
        store.index = index
        store.chunks = chunks

        if store.index.ntotal != len(store.chunks):
            raise ValueError("Index and chunks are inconsistent.")

        return store
    
    def search(self, query_embedding, top_k=3):
        """
        Find the top-k most similar chunks to a query embedding.
        """
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension does not match index.")

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Query embedding contains invalid values.")
        norm = np.linalg.norm(query_embedding)

        if not np.isclose(norm, 1.0, atol=1e-4):
            raise ValueError("Query embedding must be normalized.")

        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, k)

        # Retrieve text chunks
        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "id": int(idx)
            })

        return results

