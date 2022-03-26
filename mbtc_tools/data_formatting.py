from __future__ import annotations

import os
import re
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataFormatter:
    @staticmethod
    def read_mst_files(
        data_dir: str,
        child_only: list[str] = [
            "2018.10_BMY_Intake_MST.xlsx",
            "2018.21_ETB_Intake_MST.xlsx",
            "2018.08_NT_Intake_MST.xlsx",
        ],
    ) -> dict[str, list]:
        """Reads all MST transcripts.

        Args:
            data_dir (str): Data directory that contains MST files
            child_only (list[str], optional): Transcripts that only contain utterances of children. Defaults to [ "2018.10_BMY_Intake_MST.xlsx", "2018.21_ETB_Intake_MST.xlsx", "2018.08_NT_Intake_MST.xlsx", ].

        Returns:
            dict[str, list]: A dictionary containing filenames, dataframes for each child.
        """
        logger.info("Reading MST transcripts.")

        files = [
            filename for filename in os.listdir(data_dir) if filename.endswith("xlsx")
        ]

        all_transcripts = {"file": [], "data": [], "is_child_only": []}
        for file_name in tqdm(files):
            # Therapist or child can not be determined in these files
            data = pd.read_excel(os.path.join(data_dir, file_name))
            data = data.iloc[:, :2]  # assumes information is in first two columns

            # Only child transcript present
            if file_name in child_only:
                all_transcripts["file"].append(file_name)
                all_transcripts["data"].append(data)
                all_transcripts["is_child_only"].append(True)
            # Child and therapist transcript present
            else:
                all_transcripts["file"].append(file_name)
                all_transcripts["data"].append(data)
                all_transcripts["is_child_only"].append(False)

        return all_transcripts

    @staticmethod
    def drop_irrelevant_columns(all_transcripts: dict[str, list]) -> dict[str, list]:
        """Drops columns that do not contain transcript or identity information with
        applying a threshold to the NA ratio of a column.

        Args:
            all_transcripts (dict[str, list]): A dictionary containing filenames, dataframes for each child.

        Returns:
            dict[str, list]: A dictionary containing filenames, dataframes for each child.
        """
        logger.info("Detecting relevant columns and dropping others.")

        all_transcripts["is_single_column"] = []
        for idx, data in enumerate(tqdm(all_transcripts["data"])):
            data = data.dropna(axis=1, thresh=int(len(data.iloc[:, 0]) * 0.45)).dropna(
                axis=0
            )

            # Single column files
            if data.shape[1] == 1:
                all_transcripts["data"][idx] = data
                all_transcripts["is_single_column"].append(True)
            # Two column files
            else:
                all_transcripts["is_single_column"].append(False)
                if (
                    all_transcripts["file"][idx]
                    == "2016.06_AME_MST (PSEUDO İÇİN TEKRAR BAK).xlsx"
                ):
                    all_transcripts["data"][idx] = data.drop(columns=["Unnamed: 0"])
                else:
                    all_transcripts["data"][idx] = data

        return all_transcripts

    @staticmethod
    def format_transcripts(
        all_transcripts: dict[str, list], sep: str = "<#>"
    ) -> pd.DataFrame:
        """Matches child and therapist transcripts with regex and separates them.
        Concatenates all sentences of child or therapist with the separator.

        Args:
            all_transcripts (dict[str, list]): A dictionary containing filenames, dataframes for each child.
            sep (str, optional): Separator to join sentences with. Defaults to "<#>".

        Returns:
            pd.DataFrame: DataFrame containing child, therapist, conv. sentences and filenames
        """
        formatted = []
        for idx, data in enumerate(tqdm(all_transcripts["data"])):
            if not all_transcripts["is_child_only"][idx]:
                if not all_transcripts["is_single_column"][idx]:
                    try:
                        data.iloc[:, 0] = data.iloc[:, 0] + data.iloc[:, 1]
                    except IndexError:
                        logger.info(
                            f"Turns out {all_transcripts['file'][idx]} is actually single col."
                        )
                therapist_sents = sep.join(
                    [
                        sent[0]
                        for sent in data.values
                        if re.match(r"T\s*\:|t\s*\:", sent[0])
                    ]
                )
                child_sents = sep.join(
                    [
                        sent[0]
                        for sent in data.values
                        if re.match(
                            r"Ç\s*\:|ç\s*\:|A\s*\:|B\s*\:|E\s*\:|DS\s*\:", sent[0]
                        )
                    ]
                )
                data_formatted = {
                    "filename": all_transcripts["file"][idx],
                    "therapist_sent": therapist_sents,
                    "child_sent": child_sents,
                    "conversation_sent": sep.join(data.iloc[:, 0].tolist()),
                }
                formatted.append(data_formatted)

            else:
                therapist_sents = ""
                child_sents = sep.join([sent[0] for sent in data.values])
                conv_sents = ""
                formatted.append(
                    {
                        "filename": all_transcripts["file"][idx],
                        "therapist_sent": therapist_sents,
                        "child_sent": child_sents,
                        "conversation_sent": conv_sents,
                    }
                )

        return pd.DataFrame(formatted)

    @staticmethod
    def match_filename2id(
        mst_variables: pd.DataFrame, formatted_transcripts: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges MST variables with the formatted transcripts.

        Args:
            mst_variables (pd.DataFrame): DataFrame containing filenames and ids alongside some variables.
            formatted_transcripts (pd.DataFrame): DataFrame containing child, therapist, conv. sentences and filenames

        Returns:
            pd.DataFrame: DataFrame containing child, therapist, conv. sentences, filenames, ids, variables
        """
        file2id = {}
        mst_variables["found"] = [False for _ in range(len(mst_variables))]
        mst_variables["filename"] = [np.NaN for _ in range(len(mst_variables))]
        for idx, row in mst_variables.iterrows():
            for filename in formatted_transcripts["filename"]:
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

        not_found = [
            f
            for f in formatted_transcripts["filename"]
            if f not in mst_variables["filename"].tolist()
        ]
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
        return formatted_transcripts.merge(mst_variables, on="filename", how="outer")

    @staticmethod
    def get_labeled_files_in_dir(
        data_dir: str, file_extension: str = "xlsx"
    ) -> pd.DataFrame:
        """Reads labeled data using a directory tree. In the example below,
        file1 and file2 will be labeled as ambivalent, file3 and file4 will be labeled as secure

        Example:
        Attachment
            |_ambivalent
                |_file1
                |_file2
            |_secure
                |_file3
                |_file4

        Args:
            data_dir (str): Directory containing label directories that contain data files.
            file_extension (str, optional): Extension of files to be labeled. Defaults to "xlsx".

        Returns:
            pd.DataFrame: DataFrame containing filenames and {data_dir}_labels
        """
        labeled_files = pd.DataFrame()
        dataset_name = data_dir.split("/")[-1].lower().replace(" ", "_")
        labels = os.listdir(data_dir)

        for label in labels:
            label_dir = os.path.join(data_dir, label)
            files = [
                f for f in os.listdir(label_dir) if f.split(".")[-1] == file_extension
            ]
            labels_temp = [label for _ in range(len(files))]
            data_temp = {"filename": files, f"{dataset_name}_label": labels_temp}

            labeled_files = pd.concat([labeled_files, pd.DataFrame(data_temp)])

        return labeled_files.reset_index(drop=True)
