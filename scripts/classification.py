from __future__ import annotations

import os
import logging
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler

from arguments import (
    DataArguments,
    VectorizerArguments,
    ClassifierArguments,
    CommonArguments,
)
from mbtc_tools.vectorization import SentenceTransformersVectorizer
from mbtc_tools.utils import (
    NumpyEncoder,
    get_rule_based_keyword_detector_kwargs,
    read_formatted_dataset,
    seed_everything,
    get_mst_data_path,
)
from simple_parsing import ArgumentParser
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_validate

logger = logging.getLogger(__name__)


def main(args):
    seed_everything(args.common.seed)
    logger.info(f"Training classifier for target: {args.data.target_column}")
    train_path = get_mst_data_path(
        args.common.input_dir, args.data.target_column, "train"
    )
    test_path = get_mst_data_path(
        args.common.input_dir, args.data.target_column, "test"
    )
    keywords_path = os.path.join(args.common.input_dir, "keywords.json")

    # Check train test files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Train test files for the target {args.data.target_column} does not exist "
            f"in given input dir ({args.common.input_dir})."
            "\nExample: "
            f"\n\tTrain: {train_path}"
            f"\n\tTest: {test_path}"
        )

    # Check keywords files exist if needed to be used
    if (
        not os.path.exists(keywords_path)
        and args.vect.entity_detector is None
        and args.vect.averaging != "simple"
    ):
        raise ValueError(
            "Please provide keywords.json file in input_dir "
            f"while using {args.vect.entity_detector} entity_detector and "
            f"{args.vect.averaging} averaging."
        )

    # Read train datasets
    train = read_formatted_dataset(
        filepath=train_path,
        text_column=args.data.text_column,
        usecols=[args.data.target_column],
    )
    test = read_formatted_dataset(
        filepath=test_path,
        text_column=args.data.text_column,
        usecols=[args.data.target_column],
    )
    if args.data.shuffle:
        train = shuffle(train)
    if args.data.max_train_examples:
        train = train[: args.data.max_train_examples]
    if args.data.max_test_examples:
        test = test[: args.data.max_test_examples]

    logger.info(f"Train, test shapes: {train.shape, test.shape}")

    # Initialize vectorizer
    if args.vect.entity_detector is None:
        # Load entity detector args
        entity_detector_kwargs = get_rule_based_keyword_detector_kwargs(
            os.path.join(args.common.input_dir, "keywords.json")
        )
        vectorizer = SentenceTransformersVectorizer(
            averaging=args.vect.averaging,
            embedding_model_name=args.vect.vectorizer,
            rule_based_entity_detector_data=entity_detector_kwargs,
        )
    # Use NER model from HuggingFace Transformers
    else:
        vectorizer = SentenceTransformersVectorizer(
            averaging=args.vect.averaging,
            embedding_model_name=args.vect.vectorizer,
            entity_model_name=args.vect.entity_detector,
        )

    encoder = LabelEncoder()
    X_train = train[args.data.text_column].tolist()
    y_train = encoder.fit_transform(train[args.data.target_column].values)

    X_test = test[args.data.text_column].tolist()
    y_test = encoder.transform(test[args.data.target_column].values)

    classifier = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("scaler", StandardScaler()),
            ("classifier", SVC()),
        ]
    )
    loo = LeaveOneOut()

    scores = cross_validate(classifier, X_train, y_train, cv=loo)
    print(scores)
    print("Finished training")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    parser = ArgumentParser()
    parser.add_arguments(CommonArguments, dest="common")
    parser.add_arguments(DataArguments, dest="data")
    parser.add_arguments(VectorizerArguments, dest="vect")
    parser.add_arguments(ClassifierArguments, dest="clf")

    INPUT_DIR = "/home/emrecan/workspace/psychology-project/data"
    OUTPUT_DIR = "/home/emrecan/workspace/psychology-project/outputs"
    args = parser.parse_args(
        f"--input_dir {INPUT_DIR} --output_dir {OUTPUT_DIR} --seed 1 --averaging weighted_average --target_column attachment_label --max_train_examples 10 --max_test_examples 10 --shuffle".split()
    )
    # args = parser.parse_args()

    if not os.path.exists(args.common.output_dir):
        os.makedirs(args.common.output_dir)

    main(args)
