from __future__ import annotations
from multiprocessing import cpu_count

import os
import wandb
import logging
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sentence_transformers import SentenceTransformer

from arguments import (
    DataArguments,
    VectorizerArguments,
    ClassifierArguments,
    CommonArguments,
)
from mbtc_tools.vectorization import (
    get_input_examples_from_docs,
    initialize_averaging_model,
)
from mbtc_tools.utils import (
    get_rule_based_keyword_detector_kwargs,
    read_formatted_dataset,
    seed_everything,
    get_mst_data_path,
)
from simple_parsing import ArgumentParser
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict

logger = logging.getLogger(__name__)


def log_results_to_wandb(
    y_test,
    y_pred,
    split: str,
    classes: list[str],
):
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=classes)
    wandb.log({f"{split}_cm": cm.figure_})
    logger.info(f"{split.capitalize()} results =============================")
    clf_report = classification_report(
        y_test, y_pred, target_names=classes, output_dict=True
    )
    logger.info(pd.DataFrame(clf_report))
    wandb.log({split: clf_report})


def main(args):
    try:
        seed_everything(args.common.seed)

        config = args.clf.__dict__.copy()
        config.update(args.vect.__dict__.copy())
        if not args.common.disable_wandb:
            run = wandb.init(
                project=args.common.wandb_project,
                tags=[args.data.target_column],
                config=config,
            )

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

        encoder = LabelEncoder()
        X_train = train[args.data.text_column].tolist()
        y_train = encoder.fit_transform(train[args.data.target_column].values)

        X_test = test[args.data.text_column].tolist()
        y_test = encoder.transform(test[args.data.target_column].values)

        # Initialize vectorizer
        if args.vect.entity_detector is None:
            # Load entity detector args
            entity_detector_kwargs = get_rule_based_keyword_detector_kwargs(
                os.path.join(args.common.input_dir, "keywords.json")
            )
            averaging_model = initialize_averaging_model(
                args.vect.averaging,
                rule_based_entity_detector_data=entity_detector_kwargs,
                model_path_or_checkpoint=None,
                weight_per_entity=args.vect.weight_per_entity,
                min_weight=args.vect.min_weight,
            )
        # Use NER model from HuggingFace Transformers
        else:
            averaging_model = initialize_averaging_model(
                args.vect.averaging,
                model_path_or_checkpoint=args.vect.entity_detector,
                weight_per_entity=args.vect.weight_per_entity,
                min_weight=args.vect.min_weight,
            )

        embedding_model = SentenceTransformer(args.vect.vectorizer)
        X_train = get_input_examples_from_docs(X_train, embedding_model)
        X_test = get_input_examples_from_docs(X_test, embedding_model)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        loo = LeaveOneOut()
        classifier = Pipeline(
            steps=[
                ("averaging", averaging_model),
                # ("pca", PCA(n_components=100, random_state=args.common.seed)),
                ("scaler", StandardScaler()),
                ("classifier", SVC(random_state=args.common.seed)),
            ]
        )
        val_preds = cross_val_predict(
            classifier,
            X_train,
            y_train,
            cv=loo,
            verbose=2,
            n_jobs=cpu_count(),
        )
        if not args.common.disable_wandb:
            log_results_to_wandb(y_train, val_preds, "validation", encoder.classes_)
        if args.clf.do_test:
            logger.info("Test results =============================")
            classifier.fit(X_train, y_train)
            test_preds = classifier.predict(X_test)
            if not args.common.disable_wandb:
                log_results_to_wandb(y_test, test_preds, "test", encoder.classes_)
        if not args.common.disable_wandb:
            run.finish()

    except KeyboardInterrupt:
        logger.info("Classification manually interrupted, stopping run.")
        if not args.common.disable_wandb:
            run.finish()


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
    # args = parser.parse_args(
    #     f"--input_dir {INPUT_DIR} --output_dir {OUTPUT_DIR} --seed 1 --averaging weighted_average --target_column attachment_label ".split()
    # )
    args = parser.parse_args()

    if not os.path.exists(args.common.output_dir):
        os.makedirs(args.common.output_dir)

    main(args)
