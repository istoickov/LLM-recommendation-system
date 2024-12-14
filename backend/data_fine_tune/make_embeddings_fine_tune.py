import os
import torch
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import gc


def validate_data(data_path):
    """
    Validate and clean the data from the provided JSON file.
    """
    try:
        with open(data_path, "r") as f:
            data = [
                text for text in json.load(f) if isinstance(text, str) and len(text) > 0
            ]
        print(f"Validated data. Total samples: {len(data)}")
        return data
    except Exception as e:
        print(f"Error loading or validating data: {e}")
        return []


def generate_embeddings(model, data, batch_size=8):
    """
    Generate embeddings using a SentenceTransformer model with batching.
    """
    dataloader = DataLoader(
        data, batch_size=batch_size, num_workers=0
    )  # Disable multiprocessing
    all_embeddings = []

    for batch in tqdm(dataloader, desc="Batches"):
        try:
            # Generate embeddings
            batch_embeddings = (
                model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device="cpu",  # Force using CPU
                )
                .cpu()
                .numpy()
            )

            all_embeddings.append(batch_embeddings)

            # Clear memory after each batch to avoid memory overload
            del batch_embeddings
            gc.collect()

        except Exception as e:
            print(f"Error processing batch: {e}")
            break

    # Combine all batch embeddings
    return np.vstack(all_embeddings)


def load_model_and_generate_embeddings(model_path, data_path):
    """
    Load a fine-tuned SentenceTransformer model and generate embeddings.
    """
    # Step 1: Load the model
    print(f"Loading model from: {model_path}")
    try:
        model = SentenceTransformer(model_path, device="cpu")  # Force CPU usage
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    # Step 2: Load and validate data
    print(f"Loading and validating data from: {data_path}")
    data = validate_data(data_path)
    if not data:
        print(f"No valid data found for {model_path}. Skipping.")
        return None, None

    # Step 3: Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(model, data, batch_size=8)
    return embeddings, model


def save_embeddings(embeddings, output_path):
    """
    Save embeddings to a file for later inspection or use.
    """
    np.save(output_path, embeddings)
    print(f"Embeddings saved to: {output_path}")


def process_all_models(models_dir, data_path, output_dir):
    """
    Process all fine-tuned models in the specified directory to generate embeddings.
    """
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        model_file = os.path.join(model_path, "model.safetensors")

        if not os.path.isdir(model_path) or not os.path.isfile(model_file):
            print(f"Skipping {model_path}, missing model.safetensors file.")
            continue

        try:
            print(f"Processing model: {model_name}")
            embeddings, model = load_model_and_generate_embeddings(
                model_path, data_path
            )
            if embeddings is not None:
                output_path = os.path.join(output_dir, f"{model_name}_embeddings.npy")
                save_embeddings(embeddings, output_path)

            # Cleanup
            del embeddings, model
            gc.collect()
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")


if __name__ == "__main__":
    # Define paths
    MODELS_DIR = "./data_fine_tune"  # Directory where fine-tuned models are saved
    DATA_PATH = "./data/data_0.json"  # Path to the data used for embedding
    OUTPUT_DIR = "./embeddings"  # Directory to save embeddings

    # Process all models to generate embeddings
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_all_models(MODELS_DIR, DATA_PATH, OUTPUT_DIR)

    print("Embeddings generation completed!")
