from __future__ import annotations

import logging
import re
from functools import cached_property
from itertools import chain
from multiprocessing import cpu_count
from typing import Callable, List, Optional, Union

import spacy
from cytoolz import functoolz
from joblib import Parallel, delayed
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from .zemberek import ZemberekDocker

tqdm.pandas()

logger = logging.getLogger(__name__)
spacy.tokens.Token.set_extension("preprocessed", default=None, force=True)


class ZemberekNotInitializedError(Exception):
    """Raised when Zemberek object is not passed to a Preprocessor but
    "lemma" or "stem" are in the steps list.
    """

    pass


class Preprocessor:
    def __init__(
        self,
        steps=[
            "remove_identity_child",
            "remove_identity_therapist",
            "remove_narration",
            "lowercase",
            "normalize_i",
            "remove_number",
            "remove_punctuation",
        ],
        zemberek: Optional[ZemberekDocker] = None,
        tokenizer: Optional[Callable] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Preprocessor to apply preprocessing functions on textual
        data with given list of predefined function names or custom functions.

        Args:
            steps (list, optional): List of steps to perform on the data. All predefined functions are
                listed in name2func property of the Preprocessor object. Predefined function names
                or functions that take and return string can be used in the step list.
                Functions can also take and return spacy.tokens.Doc objects but these require
                Preprocessor.tokenize function to be added in the list beforehand.
                Defaults to [ "remove_identity_child", "remove_identity_therapist", "remove_narration", "lowercase", "normalize_i", "remove_number", "remove_punctuation", ].

            zemberek (Optional[ZemberekDocker], optional): Zemberek object to perform
                steps like pos detection, stemming, lemmatization which require morphological analysis.
                Defaults to None, meaning operations with morphology requirements will not be used.

            tokenizer (Optional[Callable], optional): Custom spaCy tokenizer.
                If no tokenizer is given, defaults to None and uses Turkish tokenizer of spaCy.

            n_jobs (Optional[int], optional): Number of cpu cores to use. Defaults to None and uses all cores.
        """

        self.steps = steps
        self.zemberek = zemberek
        self.tokenizer = tokenizer if tokenizer else spacy.blank("tr").tokenizer
        self.n_jobs = cpu_count() if not n_jobs else n_jobs

        self.filter_token = "<FILTERED>"
        self.name2func = {
            "remove_identity_therapist": self.remove_identity_therapist,
            "remove_identity_child": self.remove_identity_child,
            "remove_narration": self.remove_narration,
            "remove_number": self.remove_number,
            "remove_punctuation": self.remove_punctuation,
            "lowercase": self.lowercase,
            "normalize_i": self.normalize_i,
            "number_filter": self.number_filter,
            "punctuation_filter": self.punctuation_filter,
            "tokenize": self.tokenize,
            "detokenize": self.detokenize,
            "stem": self.stem,
            "lemma": self.lemma,
        }

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(text, str) or isinstance(text, spacy.tokens.Doc):
            return self.pipeline(text)
        elif isinstance(text, list):
            return Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
                delayed(self.pipeline)(t) for t in text
            )

    @cached_property
    def pipeline(self):
        steps = [
            self.name2func[step] if isinstance(step, str) else step
            for step in self.steps
        ]
        return functoolz.compose_left(*steps)

    def _spacy_filter(
        self, document: spacy.tokens, apply: Callable
    ) -> spacy.tokens.Doc:
        for token in document:
            token._.preprocessed = apply(token)
        return document

    def number_filter(self, document: spacy.tokens.Doc) -> spacy.tokens.Doc:
        return self._spacy_filter(
            document,
            lambda token: self.filter_token if token.like_num else token._.preprocessed,
        )

    def punctuation_filter(self, document: spacy.tokens.Doc) -> spacy.tokens.Doc:
        return self._spacy_filter(
            document,
            lambda token: self.filter_token if token.is_punct else token._.preprocessed,
        )

    def stem(self, document: spacy.tokens.Doc) -> spacy.tokens.Doc:
        if not self.zemberek:
            raise ZemberekNotInitializedError(
                "self.zemberek is not initialized, pass a ZemberekDocker"
                "object to Preprocessor to be able to use stemming"
            )
        return self._spacy_filter(
            document,
            lambda token: self.zemberek.stem(token._.preprocessed),
        )

    def lemma(self, document: spacy.tokens.Doc) -> spacy.tokens.Doc:
        if not self.zemberek:
            raise ZemberekNotInitializedError(
                "self.zemberek is not initialized, pass a ZemberekDocker"
                "object to Preprocessor to be able to use lemmatization"
            )
        return self._spacy_filter(
            document,
            lambda token: self.zemberek.lemma(token._.preprocessed),
        )

    def tokenize(self, text: str) -> spacy.tokens.Doc:
        document = self.tokenizer(text)
        return self._spacy_filter(document, lambda token: token.text)

    def detokenize(self, document: spacy.tokens.Doc) -> str:
        # this is horrendous
        return " ".join(
            " ".join(
                [
                    t._.preprocessed
                    for t in document
                    if t._.preprocessed != self.filter_token
                ]
            )
            .strip()
            .split()
        )

    def remove_narration(self, text: str) -> str:
        narration_pattern = r"(\([^)]+\.\))|(\[.+\.\])|\(.*\)|(\?\?)|(\d{2}:\d{2})"
        return re.sub(narration_pattern, "", text)

    def lowercase(self, text: str) -> str:
        text = re.sub(r"??", "i", text)
        text = re.sub(r"I", "??", text)
        text = text.lower()
        return text

    def normalize_i(self, text: str) -> str:
        return text.replace("i??", "i")

    def remove_identity_child(self, text: str) -> str:
        return re.sub(r"??\s*\:|??\s*\:|A\s*\:|B\s*\:|E\s*\:|DS\s*\:|ds\s*\:", "", text)

    def remove_identity_therapist(self, text: str) -> str:
        return re.sub(r"T\s*\:|t\s*\:", "", text)

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)

    def remove_number(self, text: str) -> str:
        return "".join(i for i in text if not i.isdigit())


def apply_preprocessing(
    dataset: pd.DataFrame,
    preprocessing_steps: list[str | Callable],
    text_column: str,
    **preprocessor_kwargs,
) -> pd.DataFrame:
    preprocessor = Preprocessor(steps=list(preprocessing_steps), **preprocessor_kwargs)
    logger.info("Preprocessing started.")
    logger.info("Before preprocessing:")
    logger.info(dataset[text_column][0][:10])
    tokenize_sentences = lambda texts: list(
        chain.from_iterable([sent_tokenize(text, "turkish") for text in texts])
    )
    logger.info("Tokenizing sentences.")
    dataset[text_column] = dataset[text_column].progress_apply(tokenize_sentences)
    logger.info("Applying preprocessing steps.")
    dataset[text_column] = dataset[text_column].progress_apply(preprocessor)
    logger.info("After preprocessing:")
    logger.info(dataset[text_column][0][:10])

    return dataset


if __name__ == "__main__":
    import pandas as pd

    zemberek = ZemberekDocker(zemberek_url="http://localhost:4567")
    # zemberek = ZemberekJava()
    # zemberek.start_jvm()
    preprocessor = Preprocessor(
        steps=[
            "remove_identity_child",
            "remove_identity_therapist",
            "remove_narration",
            "lowercase",
            "normalize_i",
            "tokenize",
            "number_filter",
            "punctuation_filter",
            "stem",
            "detokenize",
        ],
        zemberek=zemberek,
        n_jobs=4,
    )

    data = pd.read_csv(
        "/home/emrecan/workspace/psychology-project/data/data_formatted/mst_all_exploded.csv",
        nrows=1000,
    )["child_sent"].tolist()

    print(f"Before: {data[:3]}")
    preprocessed_data = preprocessor(data)
    print(f"After: {preprocessed_data[:3]}")
