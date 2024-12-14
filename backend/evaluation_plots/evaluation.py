import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from collections import defaultdict
from typing import List, Dict

from process import Process

from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK data
download("stopwords")
stop_words = set(stopwords.words("english"))


mapping_embeddings_model = {
    "faiss_index_fine_tuned_sentence-transformers-all-MiniLM-L6-v2-0_embeddings.faiss": "minilm",
    "faiss_index_fine_tuned_Muennighoff-SBERT-base-msmarco-0_embeddings.faiss": "sbert",
    "faiss_index_fine_tuned_sentence-transformers-msmarco-distilbert-base-v4-0_embeddings.faiss": "bert",
}


def evaluate_fine_tuned_models(
    query_list: List[str],
    fine_tuned_dir: str,
    data: List[Dict],
    process_obj: Process,
):
    performance = defaultdict(lambda: defaultdict(dict))

    # Directory to save the plots and JSON results
    plot_dir = "./evaluation_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Create a figure for the subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns
    axes = axes.flatten()  # Flatten axes array to index it easily

    # For each option (0 to 5), collect precision, recall, and accuracy scores for all fine-tuned models
    for option in range(0, 6):
        all_model_names = []
        all_average_precisions = []
        all_average_recalls = []

        # Loop through all .faiss files in the fine-tuned directory
        for index_file in os.listdir(fine_tuned_dir):
            print(index_file)
            if index_file.endswith(".faiss"):
                model_name = mapping_embeddings_model[index_file]
                print(
                    f"Evaluating fine-tuned model: {model_name} with option: {option}"
                )

                index_key = os.path.join(fine_tuned_dir, index_file)
                index = process_obj.faiss_client.retrive_index(index_key)
                if index is None:
                    print(
                        f"FAISS index for {model_name} with option {option} not found!"
                    )
                    continue

                precision_scores = []
                recall_scores = []
                accuracy_scores = []

                for query in query_list:
                    # Tokenize and remove stop words from query
                    query_tokens = word_tokenize(query.lower())
                    query_keywords = [
                        word
                        for word in query_tokens
                        if word not in stop_words and word not in string.punctuation
                    ]

                    # Generate embeddings for the cleaned query
                    query_cleaned = " ".join(query_keywords)
                    query_embeddings = process_obj.get_query_embeddings(
                        query_cleaned, model_name
                    )

                    # Retrieve top 10 nearest neighbors for the query
                    distances, indices = index.search(query_embeddings, 10)

                    # Generate ground truth: Match any keyword in query to data attributes
                    ground_truth = [
                        any(keyword.lower() in item for keyword in query_keywords)
                        for item in data
                    ]

                    # Get predictions for the top 10 neighbors
                    predicted_states = [ground_truth[i] for i in indices[0]]

                    # Evaluate precision, recall, and accuracy
                    ground_truth_top_10 = [
                        gt for i, gt in enumerate(ground_truth) if i in indices[0]
                    ]
                    precision = precision_score(
                        ground_truth_top_10,
                        predicted_states,
                        average="micro",
                        zero_division=1,
                    )
                    recall = recall_score(
                        ground_truth_top_10,
                        predicted_states,
                        average="micro",
                        zero_division=1,
                    )
                    accuracy = accuracy_score(ground_truth_top_10, predicted_states)

                    # Append scores
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    accuracy_scores.append(accuracy)

                # Calculate averages for all queries
                avg_precision = np.mean(precision_scores)
                avg_recall = np.mean(recall_scores)
                avg_accuracy = np.mean(accuracy_scores)

                # Store averages in performance dictionary
                performance[model_name][f"option_{option}"] = {
                    "average_precision": avg_precision,
                    "average_recall": avg_recall,
                    "average_accuracy": avg_accuracy,
                }

                all_model_names.append(model_name)
                all_average_precisions.append(avg_precision)
                all_average_recalls.append(avg_recall)

        # Plot results for this option
        ax = axes[option]
        for i, model_name in enumerate(all_model_names):
            ax.plot(
                [model_name],
                [all_average_precisions[i]],
                label=f"{model_name} Precision",
                marker="o",
            )
            ax.plot(
                [model_name],
                [all_average_recalls[i]],
                label=f"{model_name} Recall",
                marker="x",
            )
        ax.set_xlabel("Models")
        ax.set_ylabel("Average Score")
        ax.set_title(f"Option {option}")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper left")

    # Save combined plot
    plt.tight_layout()
    plot_filename = f"./{plot_dir}/all_options_comparison_fine_tuned_models.png"
    plt.savefig(plot_filename)
    plt.close()

    # Save performance to JSON file
    json_filename = f"./{plot_dir}/model_performance_fine_tuned_models.json"
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(performance, json_file, indent=4)

    return performance


def run_fine_tuned_evaluation():
    queries = [
        "Girl influencers from Germany",
        "Fashion influencers from UK",
        "Skin care influencers from USA",
        "Animal activists",
        "Sports people from USA",
        "Football influencers",
    ]

    fine_tuned_dir = "./data_fine_tune"
    process_obj = Process()
    data = json.load(open("./data/data_0.json", "r", encoding="utf-8"))

    performance = evaluate_fine_tuned_models(queries, fine_tuned_dir, data, process_obj)


if __name__ == "__main__":
    run_fine_tuned_evaluation()
