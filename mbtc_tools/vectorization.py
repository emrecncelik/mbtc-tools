from __future__ import annotations

import logging
import numpy as np

from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from weighted_bert.data import InputExample
from weighted_bert.models import WeightedAverage, WeightedRemoval
from mbtc_tools.utils import detect_keywords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class NotReadyToFit(Exception):
    """Raised when vectorizer models are not loaded but fit is called."""

    pass


class Vectorizer(ABC):
    """Common interface for various embedding methods"""

    def __init__(
        self,
        averaging: str,
        entity_model_name: str = None,
        rule_based_entity_detector_data: dict = None,
    ) -> None:
        self.averaging = averaging
        self.entity_model_name = entity_model_name
        self.rule_based_entity_detector_data = rule_based_entity_detector_data

        self.entity_detector = None
        self.embedding_model = None
        self.averaging_model = None

        if not self._is_averaging_valid():
            raise ValueError(
                f"{self.averaging} is not a valid averaging method, use one of three: "
                '"simple", "weighted_average", "weighted_removal"'
            )

        self._load_entity_detector_model(
            entity_model_name, rule_based_entity_detector_data
        )

    def _is_ready_to_fit(self):
        return self.entity_detector and self.averaging_model and self.embedding_model

    def _is_averaging_valid(self):
        return self.averaging in ["simple", "weighted_average", "weighted_removal"]

    def _init_averaging_model(
        self,
        model_path_or_checkpoint: str = None,
        rule_based_entity_detector_data: dict = None,
    ):
        if self.averaging == "simple":
            return lambda x: np.mean(x, axis=0)
        elif self.averaging == "weighted_average":
            if rule_based_entity_detector_data:
                return WeightedAverage(
                    entity_detector=self.entity_detector,
                    entity_detector_kwargs=rule_based_entity_detector_data,
                )
            elif model_path_or_checkpoint:
                raise NotImplementedError

        elif self.averaging == "weighted_removal":
            if rule_based_entity_detector_data:
                return WeightedRemoval(
                    entity_detector=self.entity_detector,
                    entity_detector_kwargs=rule_based_entity_detector_data,
                )
            elif model_path_or_checkpoint:
                raise NotImplementedError

    def _load_entity_detector_model(
        self,
        model_path_or_checkpoint: str = None,
        rule_based_entity_detector_data: dict = None,
    ) -> Vectorizer:
        if model_path_or_checkpoint:
            raise NotImplementedError
        elif rule_based_entity_detector_data:
            self.entity_detector = detect_keywords
        else:
            raise ValueError(
                "Provide model_path_or_checkpoint or rule_based_entity_detector_data."
            )

        self.averaging_model = self._init_averaging_model(
            model_path_or_checkpoint, rule_based_entity_detector_data
        )

        return self

    @abstractmethod
    def _load_embedding_model(
        self, model_path_or_checkpoint: str, **kwargs
    ) -> Vectorizer:
        pass


class SentenceTransformersVectorizer(Vectorizer, BaseEstimator, TransformerMixin):
    def __init__(
        self,
        averaging: str,
        embedding_model_name: str,
        entity_model_name: str = None,
        rule_based_entity_detector_data: dict = None,
    ) -> None:
        super().__init__(averaging, entity_model_name, rule_based_entity_detector_data)
        self.embedding_model_name = embedding_model_name
        self._load_embedding_model()

    def _load_embedding_model(self):
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self

    def fit(self, X: list[list[str]], y=None):
        if self._is_ready_to_fit():
            input_examples = [
                InputExample(doc, self.embedding_model.encode(doc)) for doc in X
            ]

            self.averaging_model = self.averaging_model.fit(input_examples)
            self.document_embeddings_ = None
        else:
            raise NotReadyToFit(
                "Call load_embedding_model and load_entity_detector_model funcs before fitting."
            )

        return self

    def transform(self, X):
        check_is_fitted(self)

        sentence_embeddings = self.embedding_model.encode(X)
        input_examples = [
            InputExample(doc, embed) for doc, embed in zip(X, sentence_embeddings)
        ]

        embeddings = self.averaging_model.transform(input_examples)
        if self.document_embeddings_ is None:
            self.document_embeddings_ = embeddings

        return embeddings


# Maybe for future experiments
class TfidfVectorizer(Vectorizer, BaseEstimator, TransformerMixin):
    pass


class FasttextVectorizer(Vectorizer, BaseEstimator, TransformerMixin):
    pass


class SimCSEVectorizer(Vectorizer, BaseEstimator, TransformerMixin):
    pass
