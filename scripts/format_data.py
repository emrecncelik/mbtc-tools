import logging
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from mbtc_tools.data_formatting import DataFormatter

logging.basicConfig(level=logging.INFO)


@dataclass
class Configuration:
    mst_dir: str
    mst_var_file: str
    attachment_dir: str
    behavior_dir: str
    output_file: Optional[str] = "mst_formatted.csv"
    sep: Optional[str] = "<#>"


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
    mst_transcripts = mst_transcripts.sort_values(by="filename")

    logging.info(f"Final shape of the dataset: {mst_transcripts.shape}")

    # Write output
    if ".csv" in args.config.output_file:
        mst_transcripts.to_csv(args.config.output_file, index=False)
    elif ".xlsx" in args.config.output_file:
        mst_transcripts.to_excel(args.config.output_file, index=False)
