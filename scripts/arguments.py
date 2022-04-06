from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

from simple_parsing import Serializable, field


@dataclass
class CommonArguments(Serializable):
    input_dir: str  # Input dir containing mst_formatted.csv, keywords.json
    output_dir: str  # Output dir for experiment outputs (models, cache etc.)
    seed: Optional[int] = None


@dataclass
class ClassifierArguments(Serializable):
    pass


@dataclass
class VectorizerArguments(Serializable):
    vectorizer: Optional[
        str
    ] = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"  # HuggingFace checkpoint or path of a SentenceTransformer model
    entity_detector: Optional[
        str
    ] = None  # HuggingFace checkpoint or path of a NER model OR 'rule_based' to use the regex based detector
    averaging: Optional[str] = field(
        choices=["simple", "weighted_average", "weighted_removal"], default="simple"
    )  # Averaging method to calculate doc embeddings. Options are 'simple', 'weighted_removal' and 'weighted_average'


@dataclass
class DataArguments(Serializable):
    target_column: str  # Target column
    text_column: Optional[str] = "child_sent"  # Text column
    n_rows: Optional[int] = None  # Number of rows to use (Num. of children)
    sep: Optional[str] = "<#>"  # Sentence separator used in formatting
    apply_sentence_tokenization: Optional[bool] = True  # Apply sentence tokenization
    max_train_examples: Optional[int] = None
    max_test_examples: Optional[int] = None
    shuffle: Optional[bool] = False
    preprocessing_steps: list[
        str
    ] = field(  # Preprocessing steps to be applied on transcripts, options are listed in Preprocessor.name2func
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


@dataclass
class VisualizerArguments(Serializable):
    pass