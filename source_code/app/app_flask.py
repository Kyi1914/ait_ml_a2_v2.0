# to control all routes

from flask import Flask, render_template, request
import carpriceprediction as a1
from a2carpriceprediction import *;

app = Flask(__name__)

# call landing page
@app.route('/')
def hello_world():
    return render_template('index.html')

# call old prediction page
@app.route('/a1_carpriceprediction')
def a1_carpriceprediction():
    return render_template('old_model_prediction.html')

# call new prediction page
@app.route('/a2_carpriceprediction')
def a2_carpriceprediction():
    return render_template('new_model_prediction.html')

# predict using old model
@app.route('/a1_predict',methods=['POST'])
def a1_predict():
    a1_input_features = [float(request.form['mileage']),
                         float (request.form['year']),
                         float(request.form['brand'])]
    prediction = a1.fn_a1_predict(a1_input_features)
    return render_template('old_model_prediction.html',prediction=prediction)

# predict using new model
@app.route('/a2_predict',methods=['POST'])
def a2_predict():
    a2_input_features = [float(request.form['mileage']),
                         float (request.form['year']),
                         float(request.form['brand'])]
    prediction = fn_a2_predict(a2_input_features)
    prediction = np.exp(prediction)
    return render_template('new_model_prediction.html',prediction=prediction)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port=80)