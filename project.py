import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import pandas as pd
import numpy as np
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from tensorflow.keras.layers import Dropout


#USE DATA COLAB FOR THIS CODE



#Import data here, must have selected and downloaded the data already to your local disk:
from google.colab import files
uploaded = files.upload()


#needs to be the name of the file
df = pd.read_csv(io.BytesIO(uploaded['PDDencoded.csv']))


#make Y for single categories
y_in =  df['Insominia']
y_sh = df['shizopherania']
y_va = df['vascula_demetia']
y_mb = df['MBD']
y_bi =  df['Bipolar']

#Make the y for the multi-classifier
y_multi = df[['Insominia', 'shizopherania', 'vascula_demetia', 'MBD', 'Bipolar' ]]


#Make X for single categories
X_in = df.drop('Insominia', axis = 1)
X_sh = df.drop('shizopherania', axis = 1)
X_va = df.drop('vascula_demetia', axis = 1)
X_mb = df.drop('MBD', axis = 1)
X_bi = df.drop('Bipolar', axis = 1)

#Make the X for the multi-classifier
X_multi = df


# Create a deep learning network with 3 hidden layers for the single classifier
def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(15,)))
    model.add(layers.Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(40, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=0.01)

    # Compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision()])

    return model

###############
#Single Classifier for Insomnia

model_in = create_model()

X = X_in
y = y_in

# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# train the model
model_in.fit(X_train, y_train, epochs = 40)


#Evaluate using the testdata set
results_in = model_in.evaluate(X_test, y_test)

################
#Single Classifier for shizopherania
model_sh = create_model()

X = X_sh
y = y_sh

# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# train the model
model_sh.fit(X_train, y_train, epochs = 40)

#Evaluate using the testdata set

results_sh = model_sh.evaluate(X_test, y_test)

############
#single classfier for vascula_demetia
model_va = create_model()

X = X_va
y = y_va

# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# train the model
model_va.fit(X_train, y_train, epochs = 40)

#Evaluate using the testdata set
results_va = model_va.evaluate(X_test, y_test)

#############
#single classfier for MBD aka ADHD

model_mb = create_model()

X = X_mb
y = y_mb

# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# train the model
model_mb.fit(X_train, y_train, epochs = 40)

#Evaluate using the testdata set

results_mb = model_mb.evaluate(X_test, y_test)
#############

#single classfier for Bipolar
model_bi = create_model()

X = X_bi
y = y_bi

# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# train the model
model_bi.fit(X_train, y_train, epochs = 40)

#Evaluate using the testdata set

results_bi = model_bi.evaluate(X_test, y_test)

###############

# Create a deep learning network with 3 hidden layers and ReLU activation for multi classifcation
def create_multi_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(16,)))
    model.add(layers.Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(40, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=0.01)

    # Compile the model

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['binary_accuracy', 'categorical_accuracy'])

    return model

###############


#Run for the multi-classifier
model_multi = create_multi_model()

X = X_multi
y = y_multi

print (X_multi)
print (y_multi)

# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# train the model
model_multi.fit(X_train, y_train, epochs = 40)

#Evaluate using the testdata set

results_multi = model_multi.evaluate(X_test, y_test)
