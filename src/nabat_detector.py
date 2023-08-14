import gc
import glob
import json
import math
import os
import pickle
import pprint
import random
import time
from collections import Counter
from pathlib import Path
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from keras import layers
from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

from src.db import NABat_DB
import src.db_handler as db_handler
import src.dataset_generator as ds


db = NABat_DB()
# Get available information for species.
species = db.query('select * from species;')

# Insert a set of predictions into the database.
def insert(data):
    db.conn.executemany(
        "insert into prediction (model_name, pulse_id, confidence, species_id) values (?,?,?,?);", data)
    db.conn.execute('commit;')


def save_model_data_splits(audio_directory):
    sample_classes = ds.get_sample_classes(audio_directory)
    species_codes = []
    for i in species:
        species_codes += [i.species_code]

    wav_directory = f'{audio_directory.split("/wav")[0]}/images'
    species_files = []
    file_counts = []
    pulse_counts = []
    x = [c.split('/')[-1] for c in glob.glob('{}/*'.format(wav_directory), recursive=True)]

    for c in x:
        if c in species_codes:
            file_count = len(glob.glob('{}/{}/*'.format(wav_directory, c), recursive=True))
            pulse_count = len(glob.glob('{}/{}/**/*'.format(wav_directory, c), recursive=True)) - file_count
            pulse_counts.append(pulse_count)
            file_counts.append(file_count)
            species_files.append(c)

    # Test the image generator function.
    gen_validate = ds.image_generator('validate')
    validate_samples_dict = dict()
    for item in gen_validate:
        image_metadata = item
        key = sample_classes[image_metadata[1]]
        if validate_samples_dict.__contains__(key):
            validate_samples_dict[key] = validate_samples_dict.get(key) + 1
        else:
            validate_samples_dict[key] = 1

    gen_test = ds.image_generator('test')
    test_samples_dict = dict()
    for item in gen_test:
        image_metadata = item
        key = sample_classes[image_metadata[1]]
        if test_samples_dict.__contains__(key):
            test_samples_dict[key] = test_samples_dict.get(key) + 1
        else:
            test_samples_dict[key] = 1

    gen_train = ds.image_generator('train')
    train_samples_dict = dict()
    for item in gen_train:
        image_metadata = item
        key = sample_classes[image_metadata[1]]
        if train_samples_dict.__contains__(key):
            train_samples_dict[key] = train_samples_dict.get(key) + 1
        else:
            train_samples_dict[key] = 1

    samples = pd.DataFrame()
    samples['Species'] = validate_samples_dict.keys()
    samples['Pulses for validation'] = validate_samples_dict.values()
    samples['Pulses for testing'] = test_samples_dict.values()
    samples['Pulses for training'] = train_samples_dict.values()
    samples = samples.set_index('Species')
    samples = samples.reindex(species_files, fill_value=0)
    samples["Pulse counts"] = pulse_counts
    samples["File counts"] = file_counts

    return samples



def train_model_round1(audio_directory, model_num):
    # Print tensorflow version and whether we have access to a gpu.
    print(tf.__version__)
    print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Set the random seed for repeatability.
    seed = 546
    random.seed(seed)

    # Create a place to store our trained models.
    try:
        os.mkdir('models')
    except:
        pass

    sample_classes = ds.get_sample_classes(audio_directory)

    # Define the batch size for the network.
    batch_size = 32

    # Create a training dataset.
    train_dataset = tf.data.Dataset.from_generator(
        ds.image_generator, args=['train'],
        output_types=({"input_1": tf.float16}, tf.int32),
        output_shapes=({"input_1": (100,100,3)}, () )                                      
        ).batch(batch_size).prefetch(1000)

    # Create a validation dataset.
    validation_dataset = tf.data.Dataset.from_generator(
        ds.image_generator, args=['validate'],
        output_types=({"input_1": tf.float16}, tf.int32),
        output_shapes=({"input_1": (100,100,3)}, () )                                      
        ).batch(batch_size).prefetch(1000)
    

    # Save the metadata for the current model run.
    with open(f'models/training_history_{model_num}.p', 'wb') as fp:

        # Define model inputs.
        inputs = layers.Input(shape=(100,100,3))

        # Define network shape.
        w = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        w = layers.Dropout(0.2)(w)
        w = layers.MaxPooling2D(padding='same')(w)
        w = layers.Conv2D(64, 3, activation='relu', padding='same')(w)
        w = layers.Conv2D(64, 3, activation='relu', padding='same')(w)
        w = layers.Dropout(0.2)(w)
        w = layers.MaxPooling2D(padding='same')(w)
        w = layers.Conv2D(128, 3, activation='relu', padding='same')(w)
        w = layers.Conv2D(128, 3, activation='relu', padding='same')(w)
        w = layers.Conv2D(128, 3, activation='relu', padding='same')(w)
        w = layers.Dropout(0.2)(w)
        w = layers.MaxPooling2D(padding='same')(w)
        w = layers.Flatten()(w)
        w = layers.Dropout(0.4)(w)
        w = layers.Dense(256, activation='relu')(w)
        w = layers.Dropout(0.4)(w)
        w = layers.Dense(256, activation='relu')(w)
        w = layers.Dropout(0.4)(w)
        w = layers.Dense(256, activation="relu")(w)
        w = layers.Dense(len(sample_classes), activation="softmax")(w)
    
        model = Model(inputs=inputs, outputs=w)
        
        # Print and plot network.
        model.summary()
        plot_model(model, to_file=f'{Path(__file__).resolve().parent}/../model_plot.png', show_shapes=True, show_layer_names=True)

        # Set the hyperparameters for this model run.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9,
                                            beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                            name='Adam'
                                            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
            )

        # Define the early stopping criteria.
        print("Define the early stopping criteria.")
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True)

        # Start the training. This will produce a preliminary (course) model.
        print("Start the training. This will produce a preliminary (course) model.")
        h = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=25,
                    verbose=1,
                    callbacks=[es]
            )

        # Save the metadata, model, and weights to disk.
        print("Save the metadata, model, and weights to disk.")
        model.save(f'{Path(__file__).resolve().parent}/../models/m-{model_num}')
        pickle.dump((h.history, sample_classes), fp)


def train_model_round2(audio_directory, model_num):
    # Print tensorflow version and whether we have access to a gpu.
    print(tf.__version__)
    print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Set the random seed for repeatability.
    seed = 546
    random.seed(seed)

    # Create a place to store our trained models.
    try:
        os.mkdir('models')
    except:
        pass

    sample_classes = ds.get_sample_classes(audio_directory)

    # Define the batch size for the network.
    batch_size = 32

    # Create a training dataset.
    train_dataset = tf.data.Dataset.from_generator(
        ds.image_generator, args=['train'],
        output_types=({"input_1": tf.float16}, tf.int32),
        output_shapes=({"input_1": (100,100,3)}, () )                                      
        ).batch(batch_size).prefetch(1000)

    # Create a validation dataset.
    validation_dataset = tf.data.Dataset.from_generator(
        ds.image_generator, args=['validate'],
        output_types=({"input_1": tf.float16}, tf.int32),
        output_shapes=({"input_1": (100,100,3)}, () )                                      
        ).batch(batch_size).prefetch(1000)
    
    # Reopen the same model for further training.
    # Here we reduce the learning rate hyperparameter by a factor of 100
    # to fine-tune the model.

    with open(f'{Path(__file__).resolve().parent}/../models/training_history_{model_num}.p', 'wb') as fp:

        model = keras.models.load_model(f'{Path(__file__).resolve().parent}/../models/m-{1}')

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001, beta_1=0.9,
                                            beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                            name='Adam'
                                            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
            )


        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True)

        h = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=5,
                    verbose=1,
                    callbacks=[es]
            )

        model.save(f'{Path(__file__).resolve().parent}/../models/m-{model_num}')
        pickle.dump((h.history, sample_classes), fp)


def train_model_round3(audio_directory, model_num):
    # Print tensorflow version and whether we have access to a gpu.
    print(tf.__version__)
    print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Set the random seed for repeatability.
    seed = 546
    random.seed(seed)

    # Create a place to store our trained models.
    try:
        os.mkdir('models')
    except:
        pass

    sample_classes = ds.get_sample_classes(audio_directory)

    # Define the batch size for the network.
    batch_size = 32

    # Create a training dataset.
    train_dataset = tf.data.Dataset.from_generator(
        ds.image_generator, args=['train'],
        output_types=({"input_1": tf.float16}, tf.int32),
        output_shapes=({"input_1": (100,100,3)}, () )                                      
        ).batch(batch_size).prefetch(1000)

    # Create a validation dataset.
    validation_dataset = tf.data.Dataset.from_generator(
        ds.image_generator, args=['validate'],
        output_types=({"input_1": tf.float16}, tf.int32),
        output_shapes=({"input_1": (100,100,3)}, () )                                      
        ).batch(batch_size).prefetch(1000)
    
    # Reopen the same model for further training.
    # Here we reduce the learning rate hyperparameter by a factor of 100
    # to fine-tune the model.

    with open(f'{Path(__file__).resolve().parent}/../models/training_history_{model_num}.p', 'wb') as fp:

        model = keras.models.load_model(f'{Path(__file__).resolve().parent}/../models/m-{2}')

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000001, beta_1=0.9,
                                            beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                            name='Adam'
                                            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
            )


        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True)

        h = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=5,
                    verbose=1,
                    callbacks=[es]
            )

        model.save(f'{Path(__file__).resolve().parent}/../models/m-{model_num}')
        pickle.dump((h.history, sample_classes), fp)


    
def run_predictions_on_provided_data(audio_directory, model_num, draw='test'):

    sample_classes = ds.get_sample_classes(audio_directory)

    # Load the tuned model from disk.
    model = keras.models.load_model(f'{Path(__file__).resolve().parent}/../models/m-{model_num}')

    # Provide a prediction for each pulse in the test set.
    to_predict1 = []
    ids = []
    data = []
    count = 0

    for p in ds.test_generator(draw):     
        count += 1
        to_predict1.append(p[0]['input_1'])
        ids.append(p[1])
        
        # Batch the predictions into groupes of 1024.
        if count != 0 and (count % 1024 == 0 ):
            predictions = model.predict(np.array(to_predict1), batch_size=1024)
            for x, prediction in enumerate(predictions):
                for i, c in enumerate(prediction):
                    data.append((str(model_num), ids[x], float(c), db_handler.get_manual_id(sample_classes[i], species)))
            
            ds.insert(data)
            to_predict1 = []
            data = []
            ids = []
            
            # Clean up
            gc.collect()
            
            # Report progress.
            print('{}'.format(int(count)))

    # Predict the remaining < 1024 predictions not batched in prior step.
    predictions = model.predict(np.array(to_predict1),batch_size=len(ids))
    for x, prediction in enumerate(predictions):
        for i, c in enumerate(prediction):
            data.append((str(model_num), ids[x], float(c), db_handler.get_manual_id(sample_classes[i], species)))
    ds.insert(data)

    db.conn.close()


# Optional, plot the accuracy and loss curves of the training and validation sets.
def plot_training():
    
    epochs_range = range(40)
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 1, 1)
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    for i in range(0,4,1):
        try:
            print(i)
            with open(f'{Path(__file__).resolve().parent}/../models/training_history_{i}.p', 'rb') as fp:
                m = pickle.load(fp)[0]
                acc = m['accuracy'] + ([0] * (epochs_range[-1] - len(m['accuracy'])))
                val_acc = m['val_accuracy'] + ([float('nan')] * (epochs_range[-1] - len(m['val_accuracy'])))
                plt.plot(epochs_range[:-1], acc, label='Training Accuracy {}'.format(i))
                plt.plot(epochs_range[:-1], val_acc, label='Validation Accuracy {}'.format(i))
    
        except Exception as e:
            pass
        
    plt.legend(loc='lower right')

        
    plt.subplot(2, 1, 2)
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    for i in range(0,4,1):
        try:
            with open(f'{Path(__file__).resolve().parent}/../models/training_history_{i}.p', 'rb') as fp:
                m = pickle.load(fp)[0]
                loss = m['loss'] + ([0] * (epochs_range[-1] - len(m['loss'])))
                val_loss = m['val_loss']
                val_loss += ([float('nan')] * (epochs_range[-1] - len(m['val_loss'])))
                
                plt.plot(epochs_range[:-1], loss, label='Training Loss {}'.format(i))
                plt.plot(epochs_range[:-1], val_loss, label='Validation Loss {}'.format(i))
    
        except Exception as e:
            pass
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()