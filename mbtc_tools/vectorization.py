from __future__ import annotations

import logging
from itertools import chain

import numpy as np
from sentence_transformers import SentenceTransformer
from weighted_bert.data import InputExample
from weighted_bert.models import WeightedAverage, WeightedRemoval
from sklearn.preprocessing import FunctionTransformer
from mbtc_tools.utils import detect_keywords

logger = logging.getLogger(__name__)


def get_input_examples_from_docs(
    docs: list[list[str]], sentence_transformer: SentenceTransformer
):
    doc_lengths = [len(doc) for doc in docs]
    sentences = list(chain.from_iterable(docs))
    sentence_embeddings = sentence_transformer.encode(sentences)
    document_embeddings = []
    prev = 0
    for idx in range(len(doc_lengths)):
        document_embeddings.append(sentence_embeddings[prev : doc_lengths[idx] + prev])
        prev += doc_lengths[idx]
    input_examples = [
        InputExample(doc, embed) for doc, embed in zip(docs, document_embeddings)
    ]
    return input_examples


def initialize_averaging_model(
    averaging_method: str,
    rule_based_entity_detector_data: dict = None,
    model_path_or_checkpoint: str = None,
    **kwargs
):
    if averaging_method == "simple":

        def simple_avg(input_examples: list[InputExample]):
            return np.array(
                [np.mean(ex.sentence_embeddings, axis=0) for ex in input_examples]
            )

        return FunctionTransformer(simple_avg)

    elif averaging_method == "weighted_average":
        if rule_based_entity_detector_data:
            return WeightedAverage(
                entity_detector=detect_keywords,
                entity_detector_kwargs=rule_based_entity_detector_data,
                **kwargs
            )
        elif model_path_or_checkpoint:
            raise NotImplementedError

    elif averaging_method == "weighted_removal":
        if rule_based_entity_detector_data:
            return WeightedRemoval(
                entity_detector=detect_keywords,
                entity_detector_kwargs=rule_based_entity_detector_data,
                **kwargs
            )
        elif model_path_or_checkpoint:
            raise NotImplementedError
    else:
        raise ValueError(
            "Invalid choice of averaging method. Choose one of simple, weighted_average or weighted_removal."
        )
