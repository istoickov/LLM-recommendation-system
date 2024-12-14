import json
import matplotlib.pyplot as plt
import numpy as np

from config import EVALUATION_RESULTS
from consts.plots import PLOT_OPTION_MAPPING


if __name__ == "__main__":
    # Load the JSON data
    data = json.load(open(EVALUATION_RESULTS, "r"))

    # Option mapping
    option_mapping = PLOT_OPTION_MAPPING

    # Extract unique options
    options = list(next(iter(data.values())).keys())
    models = list(data.keys())
    num_options = len(options)

    # Set up 3x2 grid for plots
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    axes = axes.flatten()

    # Define bar colors for precision, recall, accuracy
    colors = ["blue", "green", "red"]
    labels = ["Precision", "Recall", "Accuracy"]

    # Iterate over options and create bar charts
    for i, option in enumerate(options):
        ax = axes[i]
        precisions, recalls, accuracies = [], [], []

        # Gather metrics for all models
        for model in models:
            option_data = data[model][option]
            precisions.append(option_data["average_precision"])
            recalls.append(option_data["average_recall"])
            accuracies.append(option_data["average_accuracy"])

        # Set positions for bars
        x = np.arange(len(models))
        bar_width = 0.25

        # Plot bars for precision, recall, accuracy
        ax.bar(
            x - bar_width, precisions, width=bar_width, color=colors[0], label=labels[0]
        )
        ax.bar(x, recalls, width=bar_width, color=colors[1], label=labels[1])
        ax.bar(
            x + bar_width, accuracies, width=bar_width, color=colors[2], label=labels[2]
        )

        # Set title and labels
        ax.set_title(option_mapping[option], fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylabel("Scores", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_options, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and save
    plt.tight_layout()
    output_filename = "options_bar_chart_3x2.png"
    plt.savefig(output_filename, dpi=300)
    plt.show()

    print(f"Bar charts saved as '{output_filename}' in the same folder as this script.")
