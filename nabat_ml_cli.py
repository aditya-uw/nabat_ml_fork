"""Simple client to load tensorflow model, read a wav file, create pulse representation, and classify"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import datetime as dt

from PIL import Image

from db import NABat_DB
from prediction.prediction import Prediction
from spectrogram.spectrogram_v2 import Spectrogram

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Processor():
    def __init__(self, model, directory):
        self.directory = directory

        self.spectrogram = Spectrogram()

        self.predictor = Prediction(model, self.spectrogram.img_height,
                                    self.spectrogram.img_width, self.spectrogram.img_channels)

        self.db = NABat_DB(
            self.directory, class_list=self.predictor.CLASS_NAMES)
        self.species = self.db.query('select * from species;')
        self.species_id_lookup = [''] * 100
        for s in self.species:
            self.species_id_lookup[s.id] = s.species_code


    def process_single_wav(self, file, path):
        print("Processing: {} in path {}".format(file, path))
        spectrogram = Spectrogram()
        d = spectrogram.process_file(file)
        to_predict = ([], [], [])
        for i, m in enumerate(d.metadata):
            local_pulse_id = "pulse_"+str(m.offset)
            local_pulse_path = path+"/"+local_pulse_id+".png"
            img_obj = spectrogram.make_training_spectrogram(m.window, d.sample_rate)
            img_obj.save(local_pulse_path)
            
            disk_image = Image.open(local_pulse_path)
            img = np.array(disk_image)
            img = img[..., :3].astype('float32')
            img /= 255.0
            disk_image.close()

            to_predict[0].append(img)
            to_predict[1].append(local_pulse_id)
            to_predict[2].append(m)
            os.remove(local_pulse_path)

        all_predictions = self.predictor.predict_images(to_predict[0])

        k = 0
        dets = pd.DataFrame()
        window_offsets = []
        peak_times_of_call = []
        peak_freqs = []
        predictions = []
        prediction_scores = []
        for prediction in all_predictions:
            print("{} max class {}({}), score {}".format(to_predict[1][k], np.argmax(prediction), self.predictor.CLASS_NAMES[np.argmax(prediction)], prediction[np.argmax(prediction)]))
            print(to_predict[2][k].offset, to_predict[2][k].time)
            window_offsets+=[to_predict[2][k].offset]
            peak_times_of_call+=[to_predict[2][k].offset+to_predict[2][k].time]
            peak_freqs+=[to_predict[2][k].frequency]
            predictions+=[self.predictor.CLASS_NAMES[np.argmax(prediction)]]
            prediction_scores+=[prediction[np.argmax(prediction)]]
            k = k+1    
        dets['window_offsets_ms'] = window_offsets
        dets['peak_time_ms'] = peak_times_of_call
        dets['peak_freq_hz'] = peak_freqs
        dets['prediction'] = predictions
        dets['score'] = prediction_scores
        dets.to_csv(f'{path}/nb__{Path(file).name.split(".")[0]}.csv')

def single_run_with_ease(file, path):
    processor.process_single_wav(file, path)

def single_run_for_location(file, cfg):
    if not cfg['output_dir'].is_dir():
        cfg['output_dir'].mkdir(parents=True, exist_ok=True)
    processor.process_single_wav(file, cfg['output_dir'])

def get_params_relevant_to_data_at_location(cfg):
    data_params = dict()
    data_params['site'] = cfg['site']
    print(f"Searching for files from {cfg['site']} in {cfg['month']} {cfg['year']}")

    hard_drive_df = dd.read_csv(f'../output_dir/ubna_data_0[1|2]*', assume_missing=True, dtype=str).compute()
    if 'Unnamed: 0' in hard_drive_df.columns:
        hard_drive_df.drop(columns='Unnamed: 0', inplace=True)
    hard_drive_df["datetime_UTC"] = pd.DatetimeIndex(hard_drive_df["datetime_UTC"])
    hard_drive_df.set_index("datetime_UTC", inplace=True)
    
    files_from_location = filter_df_with_location(hard_drive_df, cfg)
    data_params['output_dir'] = cfg["output_dir"] / data_params["site"]
    print(f"Will save csv file to {data_params['output_dir']}")

    data_params['ref_audio_files'] = sorted(list(files_from_location["file_path"].apply(lambda x : Path(x)).values))
    file_status_cond = files_from_location["file_status"] == "Usable for detection"
    file_duration_cond = np.isclose(files_from_location["file_duration"].astype('float'), cfg['duration'])
    good_location_df = files_from_location.loc[file_status_cond&file_duration_cond]
    data_params['good_audio_files'] = sorted(list(good_location_df["file_path"].apply(lambda x : Path(x)).values))

    if data_params['good_audio_files'] == data_params['ref_audio_files']:
        print("All files from deployment session good!")
    else:
        print("Error files exist!")

    print(f"Will be looking at {len(data_params['good_audio_files'])} files from {data_params['site']}")

    return good_location_df, data_params


def filter_df_with_location(ubna_data_df, cfg):
    site_name_cond = ubna_data_df["site_name"] == cfg['site']
    file_year_cond = ubna_data_df.index.year == (dt.datetime.strptime(cfg['year'], '%Y')).year
    file_month_cond = ubna_data_df.index.month == (dt.datetime.strptime(cfg['month'], '%B')).month
    minute_cond = np.logical_or((ubna_data_df.index).minute == 30, (ubna_data_df.index).minute == 0)
    datetime_cond = np.logical_and((ubna_data_df.index).second == 0, minute_cond)
    file_error_cond = np.logical_and((ubna_data_df["file_duration"]!='File has no comment due to error!'), (ubna_data_df["file_duration"]!='File has no Audiomoth-related comment'))
    all_errors_cond = np.logical_and((ubna_data_df["file_duration"]!='Is empty!'), file_error_cond)
    file_date_cond = np.logical_and(file_year_cond, file_month_cond)

    filtered_location_df = ubna_data_df.loc[site_name_cond&datetime_cond&file_date_cond&all_errors_cond].sort_index()
    filtered_location_nightly_df = filtered_location_df.between_time(cfg['recording_start'], cfg['recording_end'], inclusive="left")

    return filtered_location_nightly_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process full spectrum acoustics.')

    parser.add_argument('-p, --path', dest='path', type=str, nargs=1, default=[''],
                        help='The directory to use for .wav file processing.')

    parser.add_argument('-s', "--site", type=str, help="Location to process")
    parser.add_argument('-m', "--month", type=str, help="Month to process")
    parser.add_argument('-y', "--year", type=str, help="Year to process")
    parser.add_argument('-n', "--duration", type=int, help="Duration of files to process", default=1795)
    parser.add_argument('-o', "--output_dir", type=str, help="Output directory of nb__ files")
    parser.add_argument('-k', "--recording_start", type=str, help="Recording start to take subset")
    parser.add_argument('-l', "--recording_end", type=str, help="Recording end to take subset")

    parser.add_argument('-d', "--directory", type=str, help="Path to local files to process", default='none')
    parser.add_argument('-x', "--pattern", type=str, help="Pattern in glob to look for", default='none')
    parser.add_argument("--model", type=str, help="Name of analysis model", default="m-1")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    print('Initializing...')
    args = parser.parse_args()

    processor = Processor(model=args.model, directory=args.path[0])

    if args.pattern!='none':
        wav_filepaths = list(Path(args.directory).glob(args.pattern))
        for filepath in wav_filepaths:
            print('Processing a single wav file')
            single_run_with_ease(filepath, args.path[0])
    else:
        cfg=dict()
        cfg['site']=args.site
        cfg['month']=args.month
        cfg['year']=args.year
        cfg['duration']=args.duration
        cfg["output_dir"]=Path(args.output_dir)
        cfg['recording_start']=args.recording_start
        cfg['recording_end']=args.recording_end

        good_location_df, data_params=get_params_relevant_to_data_at_location(cfg)
        for filepath in data_params['good_audio_files']:
            print('Processing a single wav file')
            single_run_for_location(filepath, cfg)
