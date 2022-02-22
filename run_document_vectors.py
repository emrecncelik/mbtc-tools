import os
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

from src.preprocessing import Preprocessor
from sentence_transformers import SentenceTransformer
from src.utils import read_formatted, generate_keyword_pattern, detect_keywords
from weighted_bert.models import WeightedAverage, WeightedRemoval

logger = logging.getLogger(__name__)


def load_data(data_dir: str, filename_prefix: str, n_rows: int = None) -> pd.DataFrame:
    dataset = pd.concat(
        [
            read_formatted(data_dir, filename)
            for filename in os.listdir(data_dir)
            if filename.startswith(filename_prefix)
        ]
    )

    if n_rows:
        dataset = dataset[:n_rows]

    logger.info(f"Transcripts with prefix '{filename_prefix}' loaded.")
    logger.info(f"\tNum. of children: {len(dataset.file.unique())}")
    logger.info(f"\tColumns: {list(dataset.columns)}")
    logger.info(f"\tShape: {dataset.shape}")
    return dataset


def preprocess(dataset: pd.DataFrame) -> dict:
    preprocessor = Preprocessor(
        steps=[
            "remove_identity_child",
            "remove_identity_therapist",
            "remove_narration",
            "lowercase",
            "normalize_i",
            # "remove_number",
            # "remove_punctuation",
            "tokenize",
            "number_filter",
            "punctuation_filter",
            "detokenize",
        ],
        n_jobs=1,  # > 1 slower since operations are cheap
    )

    logger.info("Preprocessing started.")

    dataset["child_sent"] = preprocessor(dataset["child_sent"].tolist())
    dataset_json = defaultdict(dict)
    for filename, group in dataset.groupby("file"):
        dataset_json[filename]["sentences"] = group["child_sent"].tolist()
        dataset_json[filename]["label"] = group["label"].tolist()[0]

    logger.info("Preprocessing finished.")

    return dataset_json


def compute_sentence_vectors(
    dataset: dict,
    model_checkpoint: str,
    filepath: str = "preprocessed_vectorized_data.json",
) -> dict:
    logger.info(f"Computing sentence vectors.")
    logger.info(f"\tLoading sentence transformer: {model_checkpoint}")
    embedder = SentenceTransformer(model_checkpoint)
    logger.info(f"\tModel loaded.")

    for filename in dataset.keys():
        dataset[filename]["vectors"] = embedder.encode(
            dataset[filename]["sentences"]
        ).tolist()

    logger.info(f"\tVector computation finished.")

    if filepath:
        logger.info(f"\tSaving dataset to file {filepath}")
        with open(filepath, "w") as f:
            json.dump(dataset, f, indent=4)
        logger.info(f"\tSuccessfully saved.")

    return dataset


def get_rule_based_keyword_detector_kwargs(filepath: str = "data/keywords.json"):
    with open(filepath, "r") as f:
        keywords = json.load(f)
    del keywords["cas"]  # causality removed for now

    merged_keyword_patterns, pattern2label = generate_keyword_pattern(keywords)
    entity_detector_kwargs = {
        "merged_keyword_patterns": merged_keyword_patterns,
        "pattern2label": pattern2label,
    }
    return entity_detector_kwargs


def compute_document_vectors(
    dataset: dict,
    entity_detector_kwargs: dict,
    filepath: str = "preprocessed_vectorized_data.json",
) -> dict:
    w_average = WeightedAverage(
        entity_detector=detect_keywords, entity_detector_kwargs=entity_detector_kwargs
    )
    w_removal = WeightedRemoval(
        entity_detector=detect_keywords, entity_detector_kwargs=entity_detector_kwargs
    )

    # Average embeddings
    for child in dataset.keys():
        dataset[child]["document_vector_avg"] = np.mean(
            np.array(dataset[child]["vectors"]), axis=0
        ).tolist()

    # Vectors from Weighted Average method
    for child in dataset.keys():
        dataset[child][
            "document_vector_weighted_avg"
        ] = w_average.get_document_embedding(
            document=dataset[child]["sentences"],
            sentence_embeddings=np.array(dataset[child]["vectors"]),
        ).tolist()

    # Vectors from Weighted Removal method
    documents = [dataset[child]["sentences"] for child in dataset.keys()]
    collection_sentence_embeddings = [
        np.array(dataset[child]["vectors"]) for child in dataset.keys()
    ]
    weighted_rm_vectors = w_removal.get_document_embeddings(
        documents=documents,
        collection_sentence_embeddings=collection_sentence_embeddings,
    )

    for child, embedding in zip(dataset.keys(), weighted_rm_vectors):
        dataset[child]["document_vector_weighted_rm"] = embedding.tolist()

    if filepath:
        logger.info(f"\tSaving dataset to file {filepath}")
        with open(filepath, "w") as f:
            json.dump(dataset, f, indent=4)
        logger.info(f"\tSuccessfully saved.")

    return dataset


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    DATA_DIR = "/home/emrecan/workspace/psychology-project/data/data_formatted"
    MODEL_CHECKPOINT = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    KEYWORD_DIR = "data/keywords.json"
    N_ROWS = None

    prefixes = ["attachment", "mst", "behavior"]

    embedder = None
    for p in prefixes:
        data = load_data(
            data_dir="/home/emrecan/workspace/psychology-project/data/data_formatted",
            filename_prefix=p,
            n_rows=N_ROWS,
        )
        data = preprocess(data)
        data = compute_sentence_vectors(
            data,
            model_checkpoint=MODEL_CHECKPOINT,
            filepath=f"data/{p}_vectorized.json",
        )
        entity_detector_kwargs = get_rule_based_keyword_detector_kwargs(KEYWORD_DIR)
        data = compute_document_vectors(
            data, entity_detector_kwargs, filepath=f"data/{p}_vectorized.json"
        )
