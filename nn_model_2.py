# coding: utf-8

"""
Analytics Vidya Minihack (10.02.2017)
@author: Abhishek
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization


train = pd.read_csv('train_63qYitG.csv')
test = pd.read_csv('test_XaoFywY.csv')
sample = pd.read_csv('Sample_Submission_tdRzAVW.csv')

cat_cols = ['Type_of_Cab', 'Confidence_Life_Style_Index', 'Destination_Type', 'Gender']
y = train.Surge_Pricing_Type.values

test_idx = test.Trip_ID.values
test = test.drop('Trip_ID', axis=1)
train = train.drop(['Trip_ID', 'Surge_Pricing_Type'], axis=1)

for f in cat_cols:
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


train = train.fillna(0).values
test = test.fillna(0).values

y = y - 1

ohe = preprocessing.OneHotEncoder(categorical_features=[1, 4, 5, 11], sparse=False)
ohe.fit(train)
train = ohe.transform(train)
test = ohe.transform(test)

scl = preprocessing.StandardScaler()
train = scl.fit_transform(train)
test = scl.transform(test)

y_enc = np_utils.to_categorical(y)


def nn_model():
    model = Sequential()
    print('Build model...')

    model = Sequential()
    model.add(Dense(100, input_dim=train.shape[1], init='uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = nn_model()
model.fit(train, y_enc, nb_epoch=45, batch_size=1024, verbose=2)

preds = model.predict(test)

sample.Trip_ID = test_idx
sample.Surge_Pricing_Type = np.argmax(preds, axis=1) + 1
sample.to_csv("benchmark_nn2.csv", index=False)

pd.DataFrame(preds, columns=['a', 'b', 'c']).to_csv('nn_preds_model_2.csv', index=False)

