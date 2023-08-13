from pathlib import Path
import db_handler
import nabat_detector

def run(args):

    if (args["create_spectrograms"]):
        db_handler.generate_pulses_from_dir(args['input_directory'])
    if (args["save_dataset_splits"]):
        nabat_detector.save_model_data_splits(args['input_directory']).to_csv("model_splits.csv")
    if (args["train_model0"]):
        nabat_detector.train_model_round1(args['input_directory'], "0")
    if (args["train_model1"]):
        nabat_detector.train_model_round2(args['input_directory'], "1")