from __future__ import annotations

import re


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


# if __name__ == "__main__":
#     import json
#     import pandas as pd
#     from preprocessing import Preprocessor

#     data = pd.read_csv(
#         "/home/emrecan/workspace/psychology-project/data/data_formatted/mst_all_exploded.csv",
#         nrows=100,
#     )

#     preprocessor = Preprocessor(
#         steps=[
#             "remove_identity_child",
#             "remove_identity_therapist",
#             "remove_narration",
#             "lowercase",
#             "normalize_i",
#             # "remove_number",
#             # "remove_punctuation",
#             "tokenize",
#             "number_filter",
#             "punctuation_filter",
#             "detokenize",
#         ]
#     )

#     data["child_sent"] = data["child_sent"].apply(preprocessor)

#     with open(
#         "/home/emrecan/workspace/psychology-project/data/keywords.json", "r"
#     ) as f:
#         keywords = json.load(f)

#     # TODO: causality removed for now
#     keywords.pop("cas", None)
#     sentences = []
#     labels = []

#     merged_keyword_patterns, pattern2label = generate_keyword_pattern(keywords)

#     detect_keywords(data.child_sent[:10])
#     print("hello")
