# coding: utf-8

"""
Analytics Vidya Minihack (10.02.2017)
@author: Abhishek
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing

import random


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

train = train.fillna(-999)
test = test.fillna(-999)

train = train.values
test = test.values

y = y - 1

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.025
param['max_depth'] = 12
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "merror"
param['min_child_weight'] = 1
param['subsample'] = 0.85
param['colsample_bytree'] = 0.9
param['seed'] = random.randint(0, 1000)
param['nthread'] = 20
param['missing'] = -999
param['gamma'] = 0.086
num_rounds = 1000

plst = list(param.items())

xgtest = xgb.DMatrix(test, missing=-999)

fulltrain = xgb.DMatrix(train, label=y, missing=-999)
model = xgb.train(plst, fulltrain, num_boost_round=84)
test_preds = model.predict(xgtest)


sample.Trip_ID = test_idx
sample.Surge_Pricing_Type = np.argmax(test_preds, axis=1) + 1
sample.to_csv("benchmark_xgb_1.csv", index=False)
pd.DataFrame(test_preds, columns=['a', 'b', 'c']).to_csv('xgb_preds_model_1.csv', index=False)


