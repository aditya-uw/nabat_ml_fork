# imports

# All of these modules are either included in the code base
# or provided by default on Amazon Sagemaker. 

import matplotlib.pyplot as plt
import numpy as np
import db_handler

from PIL import Image
import glob
from db import NABat_DB
import db_handler
import pprint


AUDIO_DIRECTORY = '../Downloads/data/wav'
db = NABat_DB()
# Get available information for species.
species = db.query('select * from species;')

# Return a list of files belonging to a dataset. 'train', 'test', or 'validate'.
def get_files(draw):
    result = db.query('select id, name, grts_id from file where draw = ? and grts_id != 0 order by id',(draw,))
    return result

# Return list of bat pulses that originated from a given recording file, by file id.
def get_pulses(file_id):
    result =  db.query('select * from pulse where file_id = ? order by id',(file_id,))
    return result

def get_sample_classes(directory):
    # Make sure there are at least 3 example files for each class we want to include.
    sample_classes = []
    x = [c.split('/')[-1] for c in glob.glob('{}/*'.format(directory), recursive=True)]

    for c in x:
        size = len(glob.glob('{}/{}/*'.format(directory, c), recursive=True))
        if size > 40:
            sample_classes.append(c)
            
    # Alphibitize.
    sample_classes.sort()
    return sample_classes


def initialize_available_classes(sample_classes):
    db = NABat_DB()

    # Print the classes we will include.
    pprint.pprint(sample_classes)

    # Set the database flag describing which species classes will be considered 
    # in this model training run.
    db.insert('update species set available = 0') 
    for s in sample_classes:
        db.insert('update species set available = 1 where species_code = ?',(s,)) 

    db.conn.close()

    return sample_classes


# Define a new test generator that will represent the pulse id instead
# of the species id, since we want to predict the latter.
def test_generator(draw):
    sample_classes = get_sample_classes(AUDIO_DIRECTORY)
    try:
        draw = draw.decode("utf-8")
    except:
        pass
    files = get_files(draw)
    for f in files:
        species = f.name.split('-')[0]
        if species in sample_classes:
            metadata = get_pulses(f.id)
            for i, m in enumerate(metadata):
                image = Image.open(m.path)
                img = np.array(image)
                img = img[..., :3].astype('float32')
                img /= 255.0
                image.close()
                yield {"input_1": img}, m.id


# Yield a spectrogram image and the class it belongs to. 
def image_generator(draw):
    sample_classes = get_sample_classes(AUDIO_DIRECTORY)
    try:
        draw = draw.decode("utf-8")
    except:
        pass
    
    # Get list of files.
    files = get_files(draw)
    for f in files:
        species = f.name.split('-')[0]
        if species in sample_classes:
            species_id = sample_classes.index(species)
            
            # Get a list of pulses (and path to associated spectrogram image on disk) belonging to file.
            metadata = get_pulses(f.id)
            
            for i, m in enumerate(metadata):
                # Normalize the image so that each pixel value
                # is scaled between 0 and 1.
                # print(m.path)
                image = Image.open(m.path)
                img = np.array(image)
                img = img[..., :3].astype('float32')
                img /= 255.0
                image.close()
                yield {"input_1": img}, species_id