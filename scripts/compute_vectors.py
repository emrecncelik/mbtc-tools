from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from mbtc_tools.preprocessing import apply_preprocessing
from mbtc_tools.utils import (
    NumpyEncoder,
    get_rule_based_keyword_detector_kwargs,
    read_formatted_dataset,
    seed_everything,
)
from simple_parsing import ArgumentParser, Serializable, field
from mbtc_tools.vectorization import SentenceTransformersVectorizer

logger = logging.getLogger(__name__)


@dataclass
class Configuration(Serializable):
    """Configuration for computing document embeddings"""

    input_dir: Optional[str]  # Input dir containing mst_formatted.csv, keywords.json
    output_dir: Optional[str]  # Output dir for vectorized docs and cache
    vectorizer: Optional[
        str
    ]  # HuggingFace checkpoint or path of a SentenceTransformer model
    entity_detector: Optional[
        str
    ]  # HuggingFace checkpoint or path of a NER model OR 'rule_based' to use the regex based detector
    config: Optional[str] = None  # Configuration yaml file
    averaging: Optional[str] = field(
        choices=["simple", "weighted_average", "weighted_removal"], default="simple"
    )  # Averaging method to calculate doc embeddings. Options are 'simple', 'weighted_removal' and 'weighted_average'
    text_column: str = field(
        choices=["therapist_sent", "child_sent", "conversation_sent"],
        default="child_sent",
    )  # Text column to apply preprocessing and vectorization
    sep: Optional[str] = "<#>"
    save_args: Optional[bool] = True  # Save arguments
    save_preprocessed: Optional[bool] = True  # Saves preprocessed texts
    load_preprocessed: Optional[str] = None  # Loads previously preprocessed
    apply_sentence_tokenization: Optional[bool] = True  # Apply sentence tokenization
    preprocessing_steps: list[  # Preprocessing steps to be applied on transcripts, options are listed in Preprocessor.name2func
        str
    ] = field(
        choices=[
            "remove_identity_child",
            "remove_identity_therapist",
            "remove_narration",
            "lowercase",
            "normalize_i",
            "tokenize",
            "number_filter",
            "punctuation_filter",
            "detokenize",
        ],
        default_factory=[
            "remove_identity_child",
            "remove_identity_therapist",
            "remove_narration",
            "lowercase",
            "normalize_i",
            "tokenize",
            "number_filter",
            "punctuation_filter",
            "detokenize",
        ].copy,
    )
    n_rows: Optional[int] = None  # Number of rows to use (Num. of children)
    seed: Optional[int] = 7


def main(config: Configuration):
    seed_everything(config.seed)

    input_dir_content = os.listdir(config.input_dir)
    if "mst_formatted.csv" not in input_dir_content:
        raise FileNotFoundError("Input dir should contain mst_formatted.csv file.")
    if (
        "keywords.json" not in input_dir_content
        and config.entity_detector == "rule_based"
        and config.averaging != "simple"
    ):
        raise ValueError(
            "Please provide keywords.json file in input_dir "
            f"while using {config.entity_detector} entity_detector and "
            f"{config.averaging} averaging."
        )

    if not config.load_preprocessed:
        # Read the dataset ============================================
        dataset = read_formatted_dataset(
            os.path.join(config.input_dir, "mst_formatted.csv"),
            text_column=config.text_column,
            sep=config.sep,
        )

        if config.n_rows:
            dataset = dataset[: config.n_rows]

        dataset = apply_preprocessing(
            dataset,
            preprocessing_steps=config.preprocessing_steps,
            text_column=config.text_column,
            n_jobs=1,
        )

        if config.save_preprocessed:
            logger.info("Saving preprocessed data.")
            dataset.to_json(os.path.join(config.output_dir, "preprocessed.json"))
    else:
        logger.info("Reading preprocessed dataset.")
        dataset = pd.read_json(config.load_preprocessed)
        logger.info(dataset.head())

    # Compute sentence vectors ============================================
    vectorizer = SentenceTransformersVectorizer(averaging=config.averaging)
    vectorizer.load_embedding_model(config.vectorizer)

    if config.entity_detector == "rule_based":
        entity_detector_kwargs = get_rule_based_keyword_detector_kwargs(
            os.path.join(config.input_dir, "keywords.json")
        )
        vectorizer.load_entity_detector_model(
            rule_based_entity_detector_data=entity_detector_kwargs
        )
    else:
        vectorizer.load_entity_detector_model(
            model_path_or_checkpoint=config.entity_detector
        )
    dataset["document_vector"] = vectorizer.fit_transform(
        dataset[config.text_column].tolist()
    ).tolist()

    dataset.to_json(
        os.path.join(
            config.output_dir,
            f"{config.averaging}_{config.entity_detector}_vectors.json",
        ),
        default_handler=NumpyEncoder,
    )


if __name__ == "__main__":
    input_dir = "/home/emrecan/workspace/psychology-project/data"
    output_dir = "/home/emrecan/workspace/psychology-project/outputs"
    vectorizer = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    parser = ArgumentParser()
    parser.add_arguments(Configuration, dest="config")

    args = parser.parse_args(
        f"--input_dir {input_dir} --output_dir {output_dir} --text_column child_sent --vectorizer {vectorizer} "
        "--entity_detector rule_based --averaging weighted_removal --n_rows 5".split()
    )

    # args = parser.parse_args(
    #     f"--config /home/emrecan/workspace/psychology-project/outputs/configuration.json".split()
    # )

    # args = parser.parse_args()

    if args.config.config:
        args.config = Configuration.load(args.config.config, drop_extra_fields=False)
    else:
        if (
            not args.config.input_dir
            or not args.config.output_dir
            or not args.config.vectorizer
            or not args.config.entity_detector
        ):
            raise ValueError(
                f"Provide input_dir, output_dir, vectorizer and entity_detector when not using a config file."
            )

    if not os.path.exists(args.config.output_dir):
        os.makedirs(args.config.output_dir)

    args.config.save(
        os.path.join(args.config.output_dir, "configuration.json"), indent=4
    )
    main(args.config)
