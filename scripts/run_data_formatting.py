import pandas as pd
from typing import Optional
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from simple_parsing.helpers import Serializable
from mbtc_tools.data_formatting import DataFormatter


@dataclass
class DataArguments(Serializable):
    mst_dir: str
    mst_var_file: str
    attachment_dir: str
    behavior_dir: str
    output_file: Optional[str] = "mst_formatted.csv"
    separator: Optional[str] = "<#>"
    argument_file: Optional[str] = None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(DataArguments, dest="data_args")
    args = parser.parse_args()

    args.data_args.save("formatting_args.json")

    # Read and apply initial formating on transcripts
    mst_transcripts = DataFormatter.read_mst_files(args.data_args.mst_dir)
    mst_transcripts = DataFormatter.drop_irrelevant_columns(mst_transcripts)
    mst_transcripts = DataFormatter.format_transcripts(
        mst_transcripts, args.data_args.sep
    )

    # Match mst variables with transcripts
    mst_variables = pd.read_csv(args.data_args.mst_var_file)
    mst_transcripts = DataFormatter.match_filename2id(mst_variables, mst_transcripts)

    # Match attachment and behavior labels with filenames
    attachment_labeled_files = DataFormatter.get_labeled_files_in_dir(
        args.data_args.attachment_dir
    )
    behavior_labeled_files = DataFormatter.get_labeled_files_in_dir(
        args.data_args.behavior_dir
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

    # Write output
    if ".csv" in args.data_args.output_file:
        mst_transcripts.to_csv(args.data_args.output_file, index=False)
    elif ".xlsx" in args.data_args.output_file:
        mst_transcripts.to_excel(args.data_args.output_file, index=False)
