from __future__ import annotations

import re
import os
import json
import logging
import numpy as np
import pandas as pd
from ast import literal_eval

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


def detect_keywords(
    document: list[str],
    merged_keyword_patterns: str,
    pattern2label: dict[str, str],
):
    return [
        _detect_keywords(sentence, merged_keyword_patterns, pattern2label)
        for sentence in document
    ]


def read_formatted(data_dir: str, filename: str):
    data = pd.read_csv(os.path.join(data_dir, filename), usecols=["file", "child_sent"])
    data["child_sent"] = data["child_sent"].apply(literal_eval)
    data = data.explode(column="child_sent")
    # category = filename.split("_")[0]
    label = filename.split("_")[1]
    data["label"] = [label for _ in range(len(data))]
    return data


def load_json_data(filepath: str):
    with open(filepath, "r") as f:
        dataset = json.load(f)
    return dataset
