from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import utils
tf.logging.set_verbosity(tf.logging.INFO)
#checkpoints_dir = 'checkpoints/'
# Reading the excel data
norm_data_mbc = pd.ExcelFile('F:/ISB Material/Tensorflow/Norm_Butter_Cheese.xlsx')
pred_set = pd.read_csv('F:/ISB Material/Tensorflow/pred.csv')
norm_data_MBC = norm_data_mbc.parse('sheet2')
norm_data_MBC['year'] = pd.DatetimeIndex(norm_data_MBC['Date']).year
norm_data_MBC['month'] = pd.DatetimeIndex(norm_data_MBC['Date']).month
norm_data_MBC['day'] = pd.DatetimeIndex(norm_data_MBC['Date']).day
norm_data_MBC['Organic_new1'] = np.where(norm_data_MBC['Organic']=='N', 0,1)
from sklearn.preprocessing import LabelEncoder
le12 = LabelEncoder()
norm_data_MBC['year'] = le12.fit_transform(norm_data_MBC['year'])
le13 = LabelEncoder()
norm_data_MBC['month'] = le13.fit_transform(norm_data_MBC['month'])
le14 = LabelEncoder()
norm_data_MBC['day'] = le14.fit_transform(norm_data_MBC['day'])
le15 = LabelEncoder()
norm_data_MBC['Region'] = le15.fit_transform(norm_data_MBC['Region'])
le16 = LabelEncoder()
norm_data_MBC['Pack_Size'] = le16.fit_transform(norm_data_MBC['Pack_Size'])
le17 = LabelEncoder()
norm_data_MBC['Commodity'] = le17.fit_transform(norm_data_MBC['Commodity'])
norm_data_MBC['Store_Demand_Rand_NORMDIST'] = np.log(norm_data_MBC['Store_Demand_Rand_NORMDIST'])
norm_data_MBC= norm_data_MBC.fillna(0)
from sklearn.cross_validation import train_test_split
X_MBC_norm = norm_data_MBC[['Region','Pack_Size','Commodity','Organic_new1','Low_Price','High_Price','HEB_Price_Gallon','year','month','day']]
y_MBC_norm = norm_data_MBC['Store_Demand_Rand_NORMDIST']
X_train_MBC_norm,X_test_MBC_norm,y_train_MBC_norm,y_test_MBC_norm = train_test_split(X_MBC_norm,y_MBC_norm,test_size=0.2)
train_MBC_norm = norm_data_MBC[norm_data_MBC['Date'] < pd.to_datetime('2017-01-01')]
test_MBC_norm =  norm_data_MBC[norm_data_MBC['Date'] >= pd.to_datetime('2017-01-01')]
X_train = train_MBC_norm[["Region","Pack_Size","Commodity","Organic_new1","Low_Price","High_Price","HEB_Price_Gallon","year","month","day"]]
y_train = train_MBC_norm['Store_Demand_Rand_NORMDIST']
X_test = test_MBC_norm[["Region","Pack_Size","Commodity","Organic_new1","Low_Price","High_Price","HEB_Price_Gallon","year","month","day"]]
y_test = test_MBC_norm['Store_Demand_Rand_NORMDIST']
COLUMNS = ["Region","Pack_Size","Commodity","Organic_new1","Low_Price","High_Price","HEB_Price_Gallon","year","month","day","Store_Demand_Rand_NORMDIST"]
FEATURES = ["Region","Pack_Size","Commodity","Organic_new1","Low_Price","High_Price","HEB_Price_Gallon","year","month","day"]
#COLUMNS = ["Region","Pack Size","Commodity","Organic_new1","Low_Price","High Price","year","month","day","Store Demand Rand NORMDIST"]
#FEATURES = ["Region","Pack Size","Commodity","Organic_new1","Low Price","High Price","year","month","day"]
LABEL = "Store_Demand_Rand_NORMDIST"
train = pd.concat([X_train_MBC_norm, y_train_MBC_norm], axis=1)
train.to_csv('train.csv', sep=',')
test = pd.concat([X_test,y_test], axis=1)
test.to_csv('test.csv', sep=',')
training_set = pd.read_csv("train.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)
test_set = pd.read_csv("test.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
#regressor = tf.estimator.Estimator.DNNRegressor(feature_columns=feature_cols,hidden_units=[10, 10],model_dir="/tmp/pricing_model/new")
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,hidden_units=[100, 100])
def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y = pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)
saver = tf.train.Saver()
with tf.Session() as sess:
 regressor.train(input_fn=get_input_fn(training_set), steps=600)
 ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
 loss_score = ev["loss"]
 print("Loss: {0:f}".format(sess.run(loss_score)))
 y= regressor.predict(input_fn = get_input_fn(pred_set,num_epochs = 1,shuffle = False))
 predictions = list(p["predictions"] for p in itertools.islice(y,2492))
 df = pd.Series((v[0] for v in predictions))
 sess.run(df)
 #saver.save(sess, './checkpoints/generator.ckpt')
