import os
import faiss


class FaissClient:
    def create_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        if self.index.ntotal != len(embeddings):
            print(
                f"Warning: Expected {len(embeddings)} vectors, but index contains {self.index.ntotal} vectors."
            )
        return self.index

    def save_index(self, index, index_path):
        if os.path.exists(index_path):
            os.remove(index_path)
        faiss.write_index(index, index_path)

    def retrive_index(self, index_path):
        return faiss.read_index(index_path)

    def search(self, index, query_vector, k):
        query_vector = faiss.normalize_L2(query_vector)
        return index.search(query_vector, k)
