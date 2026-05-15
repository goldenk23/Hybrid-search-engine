"""
Vector semantic search using Sentence Transformer and FAISS 
"""
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, VECTOR_INDEX_PATH

class VectorSearch:
    def __init__(self, index_path: Path | None=None):
        index_path = index_path or VECTOR_INDEX_PATH # stores the FAISS index file that contains embedding vectors and their corresponding document IDs.
        self.index_path = index_path
        self.doc_ids_path = index_path.parent / "vector_doc_ids.npy" # stores the mapping of FAISS index positions to document IDs.
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        self.index: faiss.Index | None=None # FAISS index object for efficient similarity search.
        self.doc_ids: list[str] =[] # List of document IDs corresponding to the vectors in the FAISS index.
        
    def _encode(self, texts: list[str]) ->np.ndarray:
        """ 
        converts text into normalized embedding vectors
        normalization:
        - enforcing each vector to have magnitude of 1 to ensure cosine simmilarity is equivalent to dot product in FAISS.
        cosine simmilarity:
        - two vectors in same direction(similar meaning) will have consine simmilarity close to 1, while orthogonal vectors (diffrent meaning) will have cosine simmilarity close to 0
        """
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype("float32")
    
    def build_index(self, documents: list[dict[str, Any]]) ->None:
        """ 
        Build FAISS index from documents
        - documents: list of dicts with keys 'id', 'title', 'body'
        - combines title and body for embedding
        - encodes combined text into vectors
        - adds vectors to FAISS index and saves it to disk
        """
        
        texts = [
            f"{documents.get('title', '')} {documents.get('body', '')}"
            for documents in documents
        ]
        self.doc_ids = [document.get('id', '') for document in documents]
        
        embeddings = self._encode(texts)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        """ 
        FAISS index:
        -IndexFlatIP: A simple index that performs inner product search (suitable for cosine similarity with normalized vectors).
        Observation:
        faiss index looks like:{
            postion1: vector1, # corresponds to doc_ids[0]
            postion2: vector2, # corresponds to doc_ids[1]
        }
        doc_ids looks like:[
            "doc_id1", # corresponds to position1 in FAISS index
            "doc_id2", # corresponds to position2 in FAISS index
        ]
        this is mapping between FAISS index positions and document IDs, allowing us to retrieve the original document ID after performing a search.
        """
        self.save()
        
    def save(self) ->None:
        """ Save FAISS index and document IDs to disk from RAM """
        if self.index is None:
            raise ValueError("Cannot save index: index is not built yet.")
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists ->this code creates the parent directory for the index file if it doesn't already exist.
        faiss.write_index(self.index, str(self.index_path)) # Save the FAISS index to disk using the specified path.
        np.save(self.doc_ids_path, np.array(self.doc_ids)) # Save the document IDs as a NumPy array to disk.
        
    def load(self) ->None:
        """ Load FAISS index and document IDs from disk """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Vector index not found: {self.index_path}")

        self.index = faiss.read_index(str(self.index_path))

        # Small development indexes use a sidecar position -> doc_id mapping.
        # Full MS MARCO indexes are built with faiss.IndexIDMap2, so FAISS
        # returns passage IDs directly and this sidecar file is not required.
        if self.doc_ids_path.exists():
            self.doc_ids = np.load(self.doc_ids_path).tolist()
        else:
            self.doc_ids = []
    
    def search(self, query: str, top_k: int) ->list[dict[str, Any]]:
        """ Search simmantically simmilar documents for a query"""
        
        if self.index is None:
            self.load()
        query_embedding = self._encode([query])
        
        results = []
        
        score, indices = self.index.search(query_embedding, top_k)
        """ 
        FAISS score and indices looks like:
        score: [[0.9, 0.8, 0.7]] # similarity scores for top_k results
        indices: [[0, 1, 2]] # corresponding FAISS index positions for top_k results
        so using score[0] and indices[0] we get the scores and index positions for our single query, which we can then map back to document IDs using self.doc_ids.
        """
        for score, indx_position in zip(score[0], indices[0]):
            
            if indx_position ==-1:
                continue # FAISS may return empty positions if insufficient results exist.

            if self.doc_ids:
                doc_id = self.doc_ids[indx_position]
            else:
                doc_id = str(int(indx_position))

            results.append({
                "id": doc_id,
                "score": float(score)
            })
        return results
