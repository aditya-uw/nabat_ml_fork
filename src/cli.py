from pathlib import Path

import argparse
import pipeline

def parse_args():
    """
    Defines the command line interface for the pipeline.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the dataset published by the NABat ML group"
    )
    parser.add_argument(
        "--create_spectrograms",
        action="store_true",
        help="Use to create pulses to save on disk and in database",
    )
    parser.add_argument(
        "--save_dataset_splits",
        action="store_true",
        help="Use to save dataset splits into train, test, and validation before training model",
    )
    parser.add_argument(
        "--train_model0",
        action="store_true",
        help="Use to run process to train m-0 model in round 1",
    )
    parser.add_argument(
        "--train_model1",
        action="store_true",
        help="Use to run process to train m-1 model in round 2",
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    """
    Put together important parameters and run the pipeline to generate results
    """

    args = parse_args()

    _ = pipeline.run(args)