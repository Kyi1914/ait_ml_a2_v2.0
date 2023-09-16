import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import pickle

modelname = 'source_code/app/CarPricePrediction.model'

def fn_a1_predict(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,3)
    
    #loaded model
    pickle_model = pickle.load(open(modelname, 'rb'))
    
    #take model and scaler
    model = pickle_model['model']
    scaler = pickle_model['scaler']
    
    print("loaded model")
    
    #scale the value received
    to_predict = scaler.transform(to_predict)
    
    #predict the result
    result = model.predict(to_predict)
    return np.exp(result[0])
    # return result[0]
    
print("*****  successfully called car price prediction and parse the predicted value *****")