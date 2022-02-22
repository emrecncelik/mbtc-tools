import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


logger = logging.getLogger(__name__)


def load_data(filepath: str):
    with open(filepath, "r") as f:
        dataset = json.load(f)
    return dataset


def get_tsne_embeddings(dataset: pd.DataFrame):
    logger.info("TSNE Training started.")
    logger.info("\tTraining TSNE model for weighted average embeddings.")
    dataset["weighted_avg_tsne"] = list(
        TSNE(
            n_components=2,
            perplexity=20,
            learning_rate="auto",
            init="pca",
            random_state=7,
            n_iter=3000,
        ).fit_transform(np.array(dataset["document_vector_weighted_avg"].tolist()))
    )
    logger.info("\tDone.")

    logger.info("\tTraining TSNE model for weighted removal embeddings.")
    dataset["weighted_rm_tsne"] = list(
        TSNE(
            n_components=2,
            perplexity=20,
            learning_rate="auto",
            init="pca",
            random_state=7,
            n_iter=3000,
        ).fit_transform(np.array(dataset["document_vector_weighted_rm"].tolist()))
    )
    logger.info("\tDone.")

    logger.info("\tTraining TSNE model for average embeddings.")
    dataset["average_tsne"] = list(
        TSNE(
            n_components=2,
            perplexity=20,
            learning_rate="auto",
            init="pca",
            random_state=7,
            n_iter=3000,
        ).fit_transform(np.array(dataset["document_vector_avg"].tolist()))
    )
    logger.info("\tDone.")

    return dataset


def _get_tsne_embed(dataset: pd.DataFrame, key: str, idx: int):
    return [embed[idx] for embed in dataset[key]]


def create_plots(dataset: pd.DataFrame, output: str = "figure.png"):
    plt.figure(figsize=(20, 6))
    color_count = len(dataset["label"].unique())

    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Weighted Average")
    sns.scatterplot(
        x=_get_tsne_embed(dataset, "weighted_avg_tsne", 0),
        y=_get_tsne_embed(dataset, "weighted_avg_tsne", 1),
        hue="label",
        data=dataset,
        palette=sns.color_palette("hls", color_count),
        legend="full",
        alpha=0.8,
        ax=ax1,
    )

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Weighted Removal")
    sns.scatterplot(
        x=_get_tsne_embed(dataset, "weighted_rm_tsne", 0),
        y=_get_tsne_embed(dataset, "weighted_rm_tsne", 1),
        hue="label",
        data=dataset,
        palette=sns.color_palette("hls", color_count),
        legend="full",
        alpha=0.8,
        ax=ax2,
    )

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Simple Average")
    sns.scatterplot(
        x=_get_tsne_embed(dataset, "average_tsne", 0),
        y=_get_tsne_embed(dataset, "average_tsne", 1),
        hue="label",
        data=dataset,
        palette=sns.color_palette("hls", color_count),
        legend="full",
        alpha=0.8,
        ax=ax3,
    )

    plt.savefig(output)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    prefixes = ["attachment", "behavior", "mst"]
    for p in prefixes:
        dataset = load_data(f"data/{p}_vectorized.json")
        dataset = pd.DataFrame(dataset).T
        dataset = get_tsne_embeddings(dataset)
        create_plots(dataset, f"figures/{p}_embeddings.png")
