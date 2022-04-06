from __future__ import annotations

import os
import logging
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from simple_parsing import ArgumentParser, field
from mbtc_tools.data_formatting import DataFormatter
from sklearn.model_selection import train_test_split

from mbtc_tools.utils import get_mst_data_path

logging.basicConfig(level=logging.INFO)


@dataclass
class Configuration:
    mst_dir: str
    mst_var_file: str
    attachment_dir: str
    behavior_dir: str
    target_variable: str = field(
        choices=[
            "Gen",
            "Age",
            "CB_Tot_R",
            "SSC_Tot",
            "attachment_label",
            "behavior_problems_label",
        ],
    )
    target_variable_type: str = field(choices=["categorical", "numeric"])
    test_size: float = 0.15
    min_label_frequency: int = 3
    output_dir: Optional[str] = "mst_formatted"
    sep: Optional[str] = "<#>"
    columns: list[str] = field(
        default_factory=[
            "filename",
            "therapist_sent",
            "child_sent",
            "conversation_sent",
            "ID",
            "Initials",
            # "Gen",
            # "Age",
            # "CB_Tot_R",
            # "SSC_Tot",
            # "attachment_label",
            # "behavior_problems_label",
        ].copy
    )
    all_variables: bool = False
    seed: int = 7


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_arguments(Configuration, dest="config")
    args = parser.parse_args()

    # Read and apply initial formating on transcripts
    mst_transcripts = DataFormatter.read_mst_files(args.config.mst_dir)
    mst_transcripts = DataFormatter.drop_irrelevant_columns(mst_transcripts)
    mst_transcripts = DataFormatter.format_transcripts(mst_transcripts, args.config.sep)

    # Match mst variables with transcripts
    mst_variables = pd.read_csv(args.config.mst_var_file)
    mst_transcripts = DataFormatter.match_filename2id(mst_variables, mst_transcripts)

    # Match attachment and behavior labels with filenames
    attachment_labeled_files = DataFormatter.get_labeled_files_in_dir(
        args.config.attachment_dir
    )
    behavior_labeled_files = DataFormatter.get_labeled_files_in_dir(
        args.config.behavior_dir
    )
    mst_transcripts = mst_transcripts.merge(
        attachment_labeled_files,
        on="filename",
        how="outer",
    ).merge(
        behavior_labeled_files,
        on="filename",
        how="outer",
    )
    mst_transcripts = (
        mst_transcripts.sort_values(by="filename")
        .replace(" ", pd.NA)
        .dropna(subset=["ID"])
    )
    if not args.config.all_variables:
        mst_transcripts = mst_transcripts[
            args.config.columns + [args.config.target_variable]
        ].dropna(subset=[args.config.target_variable])
        logging.info(f"Initial shape of the dataset: {mst_transcripts.shape}")

        if args.config.target_variable_type == "categorical":
            logging.info("Removing low frequency labels.")
            mst_transcripts = mst_transcripts[
                mst_transcripts.groupby(args.config.target_variable)[
                    args.config.target_variable
                ]
                .transform("count")
                .ge(args.config.min_label_frequency)
            ]
            logging.info(f"Shape of the dataset after removal: {mst_transcripts.shape}")

            train, test, _, _ = train_test_split(
                mst_transcripts,
                mst_transcripts[args.config.target_variable],
                test_size=args.config.test_size,
                stratify=mst_transcripts[args.config.target_variable],
                random_state=args.config.seed,
            )
        elif args.config.target_variable_type == "numeric":
            train, test, _, _ = train_test_split(
                mst_transcripts,
                mst_transcripts[args.config.target_variable],
                test_size=args.config.test_size,
                stratify=mst_transcripts[args.config.target_variable],
            )
        else:
            raise ValueError(
                "Target variable type must be 'numeric' or 'categorical'."
                f" Not '{args.config.target_variable_type}'"
            )

        output_path = os.path.join(
            args.config.output_dir, f"{args.config.target_variable}_formatted"
        )

        logging.info(f"Train shape: {train.shape}")
        logging.info(f"Test shape: {test.shape}")

        train.to_csv(
            get_mst_data_path(
                args.config.output_dir, args.config.target_variable, "train"
            ),
            index=False,
        )
        test.to_csv(
            get_mst_data_path(
                args.config.output_dir, args.config.target_variable, "test"
            ),
            index=False,
        )
    else:
        output_path = os.path.join(args.config.output_dir, f"mst_formatted.csv")
        mst_transcripts.to_csv(output_path, index=False)
