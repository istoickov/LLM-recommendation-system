import os
import numpy as np
import faiss
import json

from pprint import pprint
from config import RAW_DATA, DATA_FINE_TUNE_FOLDER
from consts.models import BERT, SBERT, MINILM
from process import Process
from utils.data_utils import DataUtils

# Ensure fine-tune directory exists
os.makedirs(DATA_FINE_TUNE_FOLDER, exist_ok=True)

data_utils_obj = DataUtils()
process_obj = Process()

model_embeddings_mapping = {
    BERT: "bert",
    SBERT: "sbert",
    MINILM: "minilm",
}


def print_output(indices, original_data, model_name, query, option=None):
    """Prints the output of the query.

    Args:
        indices (_type_): _description_
        original_data (_type_): _description_
        model_name (_type_): _description_
        query (_type_): _description_
        option (_type_, optional): _description_. Defaults to None.
    """
    print("-" * 70)
    print("-" * 70)
    print(f"Query: {query}")
    print(f"Model: {model_name}")
    print(f"Option: {option}")
    for index in indices[0]:
        item = original_data[index]
        pprint(
            {
                "name": item["name"],
                "country": item["state"],
                "full_name": item.get("instagram", {}).get("full_name"),
                "bio": item.get("instagram", {}).get("bio"),
                "follows": item.get("instagram", {}).get("follows"),
                "following": item.get("instagram", {}).get("following"),
                "tags": item.get("tags", []),
            }
        )
        print("-" * 50)
    print("-" * 70)
    print("-" * 70)


# Function to load embeddings and create FAISS index
def create_faiss_index(embedding_file_path, index_name):
    # Load the embeddings
    embeddings = np.load(embedding_file_path).astype(
        np.float32
    )  # FAISS requires float32 type

    # Initialize FAISS index
    dim = embeddings.shape[1]  # Embedding dimension
    faiss_index = faiss.IndexFlatL2(dim)  # Using L2 distance metric

    # Add embeddings to FAISS index
    faiss_index.add(embeddings)

    # Save the FAISS index
    faiss.write_index(faiss_index, index_name)
    print(f"FAISS index saved as {index_name}")

    return faiss_index


def main(query_list):
    # Loop through all embedding files and create corresponding FAISS indexes
    for embedding_file in os.listdir(DATA_FINE_TUNE_FOLDER):
        if embedding_file.endswith(".npy"):
            embedding_name = embedding_file[:-4]  # Remove '.npy' extension
            embedding_file_path = os.path.join(DATA_FINE_TUNE_FOLDER, embedding_file)

            # Define the name for the FAISS index
            index_name = os.path.join(
                DATA_FINE_TUNE_FOLDER, f"faiss_index_fine_tuned_{embedding_name}.faiss"
            )

            # Create and save the FAISS index
            index = create_faiss_index(embedding_file_path, index_name)

        model_name = model_embeddings_mapping[embedding_name]
        option = 0
        data = json.load(open(RAW_DATA, "r", encoding="utf-8"))
        for query in query_list:
            print(f"Query: {query}")
            query = data_utils_obj.clean_summary(query)

            query_embeddings = process_obj.get_query_embeddings(query, model_name)

            print("Query embedding shape: ", query_embeddings.shape)
            distances, indices = index.search(query_embeddings, 10)
            print("Distances: ", distances)
            print("Indices: ", indices)

            print_output(indices.tolist(), data, model_name, query, option)


if __name__ == "__main__":
    queries = [
        "Girl influencers based in the Germany",
        "Fashion influencers based in the America",
        "Skin care influencers based in the United Kingdom",
        "Animal activists based in UK",
        "e-Sports people from USA",
    ]

    main(queries)
