'''
@ author: Priyanka Shrestha
* Deployment code for a convolutional neural network (CNN) predicting guide scores for Cas13 sgRNA 
* based on nucleotide sequence.
* Command to run the API is uvicorn model_deployment:app --reload
* Add /docs to the end of host to use API
'''
# imports
import uvicorn
import pathlib
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from helper_functions import *
from pydantic import BaseModel
import tensorflow as tf
from typing import Optional

# 
BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "guidescore_model.h5"
MODEL = None

# class for data -- not used
class Sequences(BaseModel):
    def __init__(self):
        self.sequences = []

# create app object
app = FastAPI()
#pickle_in = open("cas13_guidescore.pkl", 'r')
#model = pickle.load(pickle_in)

@app.on_event("startup")
def on_startup():
    # load_model
    global MODEL
    if MODEL_PATH.exists():
        MODEL = tf.keras.models.load_model(MODEL_PATH)

def predict(query:str):
    sequences = prepare_rna(query)
    x_input = np.array(sequences)
    print(x_input)
    y_output = MODEL.predict(x_input)
    y_output = y_output.flatten()
    labeled_preds = [{f"{query[i]}": y_output[i]} for i in range(len(query))]
    print(labeled_preds)

# expose prediction functionality, make prediction 
@app.post('/')
def read_index(q):
    y_output = predict(q)
    y_output = y_output.flatten()
    labeled_preds = [{f"{q[i]}": y_output[i]} for i in range(len(q))]
    return labeled_preds

'''
# index
@app.get('/')
def read_index(q:Optional[str] = None):
    global MODEL
    print(MODEL)
    return {"message" : "Hello World", "BASE_DIR" : str(BASE_DIR)}


# route with a single parameter, returns parameter within a message
@app.get('./{name}')
def get_name(name: str):
    return {'message' : f'Hello, {name}'}
'''

# Run API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


