from tensorflow import keras
import numpy as np
import gc

from db import NABat_DB
import db_handler
import dataset_generator as ds


db = NABat_DB()
# Get available information for species.
species = db.query('select * from species;')

# Insert a set of predictions into the database.
def insert(data):
    db.conn.executemany(
        "insert into prediction (model_name, pulse_id, confidence, species_id) values (?,?,?,?);", data)
    db.conn.execute('commit;')

    
def run_predictions_on_provided_data(m, sample_classes, draw='test'):

    # Load the tuned model from disk.
    model = keras.models.load_model(f'models/m-{m}')

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
                    data.append((str(m), ids[x], float(c), db_handler.get_manual_id(sample_classes[i], species)))
            
            insert(data)
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
            data.append((str(m), ids[x], float(c), db_handler.get_manual_id(sample_classes[i], species)))
    insert(data)

    db.conn.close()