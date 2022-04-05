from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional
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
    ] = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"  # HuggingFace checkpoint or path of a SentenceTransformer model
    entity_detector: Optional[
        str
    ] = None  # HuggingFace checkpoint or path of a NER model OR 'rule_based' to use the regex based detector
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
        and config.entity_detector is None
        and config.averaging != "simple"
    ):
        raise ValueError(
            "Please provide keywords.json file in input_dir "
            f"while using {config.entity_detector} entity_detector and "
            f"{config.averaging} averaging."
        )

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

    # Compute sentence vectors ============================================
    # Use rule based keyword detector
    if config.entity_detector is None:
        entity_detector_kwargs = get_rule_based_keyword_detector_kwargs(
            os.path.join(config.input_dir, "keywords.json")
        )

        vectorizer = SentenceTransformersVectorizer(
            averaging=config.averaging,
            embedding_model_name=config.vectorizer,
            rule_based_entity_detector_data=entity_detector_kwargs,
        )
    # Use NER model from HuggingFace Transformers
    else:
        vectorizer = SentenceTransformersVectorizer(
            averaging=config.averaging,
            embedding_model_name=config.vectorizer,
            entity_model_name=config.entity_detector,
        )

    # Train vectorizer, transform text
    dataset["document_vector"] = vectorizer.fit_transform(
        dataset[config.text_column].tolist()
    ).tolist()

    # Save computed vectors
    dataset.to_json(
        os.path.join(
            config.output_dir,
            f"{config.averaging}_vectors.json",
        ),
        default_handler=NumpyEncoder,
    )


if __name__ == "__main__":
    input_dir = "/home/emrecan/workspace/psychology-project/data"
    output_dir = "/home/emrecan/workspace/psychology-project/outputs"

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    parser = ArgumentParser()
    parser.add_arguments(Configuration, dest="config")

    args = parser.parse_args(
        f"--input_dir {input_dir} --output_dir {output_dir} --text_column child_sent "
        "--averaging weighted_removal --n_rows 5".split()
    )

    # args = parser.parse_args(
    #     f"--config /home/emrecan/workspace/psychology-project/outputs/configuration.json".split()
    # )

    # args = parser.parse_args()

    if args.config.config:
        args.config = Configuration.load(args.config.config, drop_extra_fields=False)
    else:
        if not args.config.input_dir or not args.config.output_dir:
            raise ValueError(
                f"Provide input_dir, output_dir when not using a config file."
            )

    if not os.path.exists(args.config.output_dir):
        os.makedirs(args.config.output_dir)

    args.config.save(
        os.path.join(args.config.output_dir, "configuration.json"), indent=4
    )
    main(args.config)
