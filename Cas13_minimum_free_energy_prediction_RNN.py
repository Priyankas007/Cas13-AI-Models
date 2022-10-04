import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn import preprocessing
import neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor
import csv
from kerastuner.tuners import Hyperband
import numpy as np
import pandas as pd

#import and clean up data
sequences = []
scores = []
files = ['train.tsv', 'test.tsv', 'val.tsv']

for item in files:
        with open(item) as tsv_file:
                tsv_reader = csv.reader(tsv_file, delimiter='\t')
                for line in tsv_reader:
                        sequences.append(line[0])
                        scores.append(line[1])
                tsv_file.close()

#create a dataFrame with the sequences and corresponding MFE
combined = list(zip(sequences, scores))
df = pd.DataFrame(combined, columns = ['Sequences', 'MFE'])

#conver the MFE's from a string type to float type
df['MFE'] = df['MFE'].astype(float)

#MFE statistics
print("The mean of the MFE's is", df['MFE'].mean())
print("The standard deviation of the MFE's is", df['MFE'].std())
print("The min of the MFE's is", df['MFE'].min())
print("The max of the MFE's is", df['MFE'].max())

#boxplot of MFE
df['MFE'].plot.box()
plt.savefig('mfe_boxplot.png')

#histogram of MFE
box = df['MFE'].plot.hist()
plt.savefig('mfe_histogram.png')

#function to encode RNA into vectors
def encodeRNA(sequences):
    encoded_rna = []
    for char in sequences:
        if(char == 'A'):
            encoded_rna.append([1,0,0,0,0,0])
        elif(char == 'C'):
            encoded_rna.append([0,1,0,0,0,0])
        elif(char == 'G'):
            encoded_rna.append([0,0,1,0,0,0])
        elif(char == 'U'):
            encoded_rna.append([0,0,0,1,0,0])
        elif(char == 'T'):
            encoded_rna.append([0,0,0,0,1,0])
        else:
            encoded_rna.append([0,0,0,0,0,1])
    return encoded_rna

df['encoded_RNA'] = '';
for i in range (len(df['Sequences'])):
    df['encoded_RNA'][i] = encodeRNA(df['Sequences'][i])

#split data into training, validation, and testing sets
x_test, y_train, y_test = train_test_split(df['encoded_RNA'], df['MFE'], test_size = 0.1, random_state = 4)


#convert lists into numpy arrays and shape them into have dimension
#(batch_size, timesteps, features)
x_train = sequence.pad_sequences(x_train, dtype=float)
x_test = sequence.pad_sequences(x_test, dtype=float)
x_val = sequence.pad_sequences(x_val, dtype=float)

#build a hypermodel using the model builder function
def model_builder(hp):
    #define set of hyperparameters for tuning and the range of values for each
    neurons = hp.Int('neurons', min_value = 50, max_value = 300, step = 20)
    hidden_layers = hp.Int('hidden_layers', min_value=2, max_value=10, step = 2)
    dropout = hp.Choice('dropout', values = [0.2, 0.3, 0.4, 0.5, 0.6])
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    activation = hp.Choice('activiation', values=['tanh', 'sigmoid', 'relu'])
    
    #define model
    model = Sequential()
    model.add(LSTM(units=neurons, activation=activation, return_sequences=True, 
                   input_shape = [23,6]))
    
    for i in range(hidden_layers):
        model.add(LSTM(units=hp.Int('neurons_' + str(i),
                                     min_value = 50,
                                     max_value = 300,
                                     step = 20),
                                     activation=activation, 
                                     return_sequences=True,
                                     dropout=dropout))
    model.add(Bidirectional(LSTM(units=neurons, activation=activation)))
    model.add(Dense(1))
    
    model.compile (
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss = "mean_squared_error",
        metrics = ["mean_squared_error"]
        
    )
    
    return model

tuner = Hyperband(model_builder,
                    objective='val_loss',
                    max_epochs=10,
                    executions_per_trial=3,
                    directory='test',
                    project_name='mfe_test')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode = 'min')

tuner.search(x_train,y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")


#built the model with the optimal hyperparameters and train it on the data for 100 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)
val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#retrain the model
final = hypermodel.fit(x_train, y_train, epochs=best_epoch)

#fit and plot the mode
train_mse = hypermodel.evaluate(x_train, y_train, verbose=0)
valid_mse = hypermodel.evaluate(x_valid, y_valid, verbose=0)
print('Train: ', train_mse)
print('Valid: ', valid_mse)

# plot loss during training
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(final.final['loss'], label='train')
pyplot.plot(final.final['val_loss'], label='valid')
pyplot.legend()
pyplot.ylim([0,4])
pyplot.savefig('final_model_loss.png')

#plot predictions versus actual MFE
pyplt.title('Predictions versus actual MFE')
pyplt.plot(final.predict(x_train))
pyplt.plot(y_train)
pyplt.legend()
pyplt.savefig('predictions.png')


