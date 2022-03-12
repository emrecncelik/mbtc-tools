from __future__ import annotations

import re
import os
import json
import logging
import numpy as np
import pandas as pd
from ast import literal_eval

logger = logging.getLogger(__name__)


def match_file2id(mst_variables: pd.DataFrame, filenames: list[str]):
    file2id = {}
    mst_variables["found"] = [False for _ in range(len(mst_variables))]
    mst_variables["filename"] = [np.NaN for _ in range(len(mst_variables))]
    for idx, row in mst_variables.iterrows():
        for filename in filenames:
            filename_clean = (
                filename.replace("MST", "")
                .replace("xlsx", "")
                .replace("_", "")
                .replace(".", "")
            )
            if str(row["ID"]) in filename_clean:
                file2id[filename] = f'{str(row["ID"])}_{row["Initials"]}'
                mst_variables.loc[idx, "found"] = True
                mst_variables.loc[idx, "filename"] = filename
                logger.info(
                    f"Matched: {filename} | ID_Initials: {row['ID']}_{row['Initials']}"
                )

    not_found = [f for f in filenames if f not in mst_variables["filename"].tolist()]
    logger.info(f"Initial list of files not matched: {not_found}")

    for idx, row in mst_variables[mst_variables.isna()["filename"]].iterrows():
        for filename in not_found:
            if row["Initials"] in filename:
                mst_variables.loc[idx, "found"] = True
                mst_variables.loc[idx, "filename"] = filename
                not_found.remove(filename)
                logger.info(
                    f"Matched: {filename} | ID_Initials: {row['ID']}{row['Initials']}"
                )

    del mst_variables["found"]
    logger.info(f"Final list of files not matched: {not_found}")
    return mst_variables


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


def get_labeled_files_in_dir(data_dir: str, file_extension: str = "xlsx"):
    labeled_files = pd.DataFrame()
    dataset_name = data_dir.split("/")[-1].lower().replace(" ", "_")
    labels = os.listdir(data_dir)

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        files = [f for f in os.listdir(label_dir) if f.split(".")[-1] == file_extension]
        labels_temp = [label for _ in range(len(files))]
        data_temp = {"filename": files, f"{dataset_name}_label": labels_temp}

        labeled_files = pd.concat([labeled_files, pd.DataFrame(data_temp)])

    return labeled_files.reset_index(drop=True)


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
