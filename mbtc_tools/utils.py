from __future__ import annotations

import re
import os
import json
import logging
import pandas as pd
import numpy as np
import random
import torch

logger = logging.getLogger(__name__)


def generate_keyword_pattern(keywords: dict[str, list[str]]):
    pattern2label = {}
    for key in keywords:
        keywords[key] = [w.replace("mek", "").replace("mak", "") for w in keywords[key]]
        pattern2label[r"\w+|".join(keywords[key]) + "\w+"] = key

    merged_keyword_patterns = "|".join(pattern2label.keys())

    return merged_keyword_patterns, pattern2label


def _detect_keywords(
    sentence: str,
    merged_keyword_patterns: str,
    pattern2label: dict[str, str],
):
    labels = []

    def append_to_label(match, type_):
        labels.append(
            {
                "text": match.group(),
                "start": match.span()[0],
                "end": match.span()[0] + len(match.group()),
                "type": type_,
            }
        )

    for match in re.finditer(f"{merged_keyword_patterns}|\w+", sentence):
        if re.findall(merged_keyword_patterns, match.group()):
            if re.findall(list(pattern2label.keys())[0], match.group()):
                append_to_label(match, "c1-pos")
            elif re.findall(list(pattern2label.keys())[1], match.group()):
                append_to_label(match, "c1-neg")
            elif re.findall(list(pattern2label.keys())[2], match.group()):
                append_to_label(match, "c2")
            elif re.findall(list(pattern2label.keys())[3], match.group()):
                append_to_label(match, "c3")
            elif re.findall(list(pattern2label.keys())[4], match.group()):
                append_to_label(match, "c4")
            elif re.findall(list(pattern2label.keys())[5], match.group()):
                append_to_label(match, "c5")
            elif re.findall(list(pattern2label.keys())[6], match.group()):
                append_to_label(match, "c9")
            # elif  re.findall(list(keywords_regex.keys())[7], match.group()): labels.append('cas')
    return labels


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


def detect_keywords(
    document: list[str],
    merged_keyword_patterns: str,
    pattern2label: dict[str, str],
):
    return [
        _detect_keywords(sentence, merged_keyword_patterns, pattern2label)
        for sentence in document
    ]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_formatted_dataset(
    filepath: str, text_column: str, usecols: list[str] = [], sep="<#>", **kwargs
):
    usecols.extend(["filename", text_column])
    if ".xlsx" in filepath:
        dataset = pd.read_excel(filepath, usecols=usecols, **kwargs)
    elif ".csv" in filepath:
        dataset = pd.read_csv(filepath, usecols=usecols, **kwargs)
    else:
        raise ValueError(
            f"Extension {filepath.split('.')[-1]} is not supported. Use csv or xlsx."
        )
    dataset[text_column] = dataset[text_column].apply(lambda x: x.split(sep))
    return dataset


def load_json_data(filepath: str):
    with open(filepath, "r") as f:
        dataset = json.load(f)
    return dataset


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
