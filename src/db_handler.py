# imports

# With the exception of the librosa library installed above, all of these modules are 
# either included in the code base or provided by default on Amazon Sagemaker. 

import gc
import glob
import multiprocessing as mp
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from src.db import NABat_DB
from src.spectrogram import Spectrogram
    
SPECTROGRAM_LOCATION = '../../Downloads/data/images'

# Given a species code, return a numeric id.
def get_manual_id(species_code, species):
    for s in species:
        if s.species_code == species_code:
            return s.id


# This method is meant to be called in parallel and will take a single file path
# and produce a spectrogram for each pulse detected using a BLED within the recording.
def process_file(file, db_name="db0"):
    # Randomly and proprotionally assign files to the train, validate, and test sets.
    # 80% train, 10% validate, 10% test
    draw = None
    r = random.random()
    if r < 0.80:
        draw = 'train'
    elif r < 0.90:
        draw = 'test'
    else:
        draw = 'validate'
      

    db = NABat_DB(p=db_name)
    species = db.query('select * from species;')
    
    # Get metadata about the recording from the file name.
    species_code = file.split('/')[-2]
    manual_id = get_manual_id(species_code, species)
    grts_id = file.split('-')[1].split('.')[0]
    file_name = f'{species_code}-{grts_id}'

    file_path = Path('{}/{}/{}'.format(SPECTROGRAM_LOCATION, species_code, file_name))
    file_path.mkdir(parents=True, exist_ok=True)

    # Process file and return pulse metadata.
    spectrogram = Spectrogram()
    d = spectrogram.process_file(file)

    # Add the file to the database.
    file_id, draw = db.add_file(
                    file_name, d.duration, d.sample_rate, manual_id, grts_id, draw=draw)

    # For each pulse within file...
    for i, m in enumerate(d.metadata):
        # ...create a place to put the spectrogram.
        path = '{}/{}/{}/t_{}.png'.format(SPECTROGRAM_LOCATION, species_code, file_name, m.offset)
        
        # Add the pulse to the database.
        pulse_id = db.add_pulse(file_id, m.frequency,
                                m.amplitude, m.snr, m.offset, m.time, None, path)
        # On success...
        if pulse_id:
            # ...create a spectrogram image surrounding the pulse and save to disk.
            # If the image already exists on disk, skip this process because very time-consuming to save.
            if (not(Path(path).is_file())):
                img = spectrogram.make_spectrogram(m.window, d.sample_rate)
                img.save(path)
                img.close()
            
    # Close the database connection.
    db.conn.close()


def process_input_file(file, db_name="db1"):
    # Randomly and proprotionally assign files to the train, validate, and test sets.
    # 80% train, 10% validate, 10% test
    draw = None
    r = random.random()
    if r < 0.80:
        draw = 'train'
    elif r < 0.90:
        draw = 'test'
    else:
        draw = 'validate'
      

    db = NABat_DB(p=db_name)
    file_name = Path(file).name

    file_path = Path('{}/{}'.format(SPECTROGRAM_LOCATION, file_name))
    file_path.mkdir(parents=True, exist_ok=True)

    # Process file and return pulse metadata.
    spectrogram = Spectrogram()
    d = spectrogram.process_file(file)

    # Add the file to the database.
    file_id, draw = db.add_file(
                    file_name, d.duration, d.sample_rate, -1, 62145, draw=draw)

    # For each pulse within file...
    for i, m in enumerate(d.metadata):
        # ...create a place to put the spectrogram.
        path = '{}/{}/t_{}.png'.format(SPECTROGRAM_LOCATION, file_name, m.offset)
        
        # Add the pulse to the database.
        pulse_id = db.add_pulse(file_id, m.frequency,
                                m.amplitude, m.snr, m.offset, m.time, None, path)
        # On success...
        if pulse_id:
            # ...create a spectrogram image surrounding the pulse and save to disk.
            # If the image already exists on disk, skip this process because very time-consuming to save.
            if (not(Path(path).is_file())):
                img = spectrogram.make_spectrogram(m.window, d.sample_rate)
                img.save(path)
                img.close()
            
    # Close the database connection.
    db.conn.close()


def generate_pulses_from_dir(directory):
    # Use as many threads as we can, leaving one available to keep notebook responsive.
    thread_count = (mp.cpu_count() - 1)
    print('using {} threads'.format(thread_count))

    # Gather wav files.
    files = glob.glob('{}/**/*.wav'.format(directory), recursive=True)
    progress = np.ceil(len(files) * 0.01).astype('int')
    print(progress, files)

    # Start the creation process in parallel and report progress.
    for i in range(0,len(files),progress):
        with Pool(thread_count) as p:
            p.map(process_file, files[i:i+progress])
            gc.collect()
            print('{}%'.format(int(i/progress)))


def delete_files_from_db(db_name="db0", condition=''):
    db = NABat_DB(p=db_name)
    if condition != '':
        db.cursor.execute(f"DELETE FROM file WHERE {condition};")
        db.conn.execute('commit;')
    else:
        db.cursor.execute("DELETE FROM file;")
        db.conn.execute('commit;')        
    # Close the database connection.
    db.conn.close()


def delete_pulses_from_db(db_name="db0", condition=''):
    db = NABat_DB(p=db_name)
    if condition != '':
        db.cursor.execute(f"DELETE FROM pulse WHERE {condition};")
        db.conn.execute('commit;')
    else:
        db.cursor.execute("DELETE FROM pulse;")
        db.conn.execute('commit;')        
    # Close the database connection.
    db.conn.close()


def delete_predictions_from_db(db_name="db0", condition=''):
    db = NABat_DB(p=db_name)
    if condition != '':
        db.cursor.execute(f"DELETE FROM prediction WHERE {condition};")
        db.conn.execute('commit;')
    else:
        db.cursor.execute("DELETE FROM prediction;")
        db.conn.execute('commit;')        
    # Close the database connection.
    db.conn.close()