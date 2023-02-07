import pre_processing
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import joblib
from lightgbm import LGBMClassifier
import os

training = True # whether to train a prediction model
hardware_aware = False # whether to train a model that work accross different GPUs (RTX 2080, RTX 3090, Tesla V100)
                       # if False, we train a model specialized for RTX 3090

Model_Exists = os.path.exists("model")
if not Model_Exists: os.mkdir("model")


if __name__ == "__main__":

   if hardware_aware:
      data_path= f'./data-hardware-aware/' 
   else:
      data_path= f'./data/' 

   # pre-processing the data
   if training:
      columnNames1,rows1 = pre_processing.load_data(data_path+f'train.csv')   # train the model
      columnNames1,x_train,y_train = pre_processing.separate(columnNames1, rows1,'tag')
   
   columnNames2,rows2 = pre_processing.load_data(data_path+f'test.csv')    # test the model
   columnNames3,rows3 = pre_processing.load_data(data_path+f'data.csv')    # to evaluate the model performance 
   columnNames2,x_test,y_test = pre_processing.separate(columnNames2, rows2,'tag')
   columnNames3,x_data,y_data = pre_processing.separate(columnNames3, rows3,'maxval')

   if training:
      model = LGBMClassifier(n_estimators=1000) 
      model.fit(x_train, y_train)

      #save the model
      joblib.dump(model, f'./model/model.txt')
   
   else: # load pre-trained model
      model = joblib.load(f'./model/model.txt')

   # make the prediction
   y_predict = model.predict(x_test)

   # print("predict tag :", y_predict)
   # print("real tag:", y_test)

   # evaluate prediction accuracy
   predictions = [round(value) for value in y_predict]
   accuracy = accuracy_score(y_test, predictions)
   print("Model accuracy: %.2f%%" % (accuracy * 100.0))

   # compute normalized model performance
   y_rate = np.zeros_like(y_test)
   for i in range(0, y_rate.shape[0]):
      y_rate[i] = x_data[i][int(y_predict[i])]/y_data[i]
   print('Nomalized model performance: %.2f%%' % (pow(y_rate.prod(),1/y_rate.shape[0])*100))


