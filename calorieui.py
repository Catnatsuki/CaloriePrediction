from flask import Flask, render_template, request, jsonify
import pickle
import os
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__, static_folder='static')

dir = os.getcwd()
print(dir)

calories_model = pickle.load(open('calories_plr.sav', 'rb'))

def predictions(gender, duration, heart_rate):
    gender_numerical = 0
    if gender.casefold() == 'male':
        gender_numerical = 0
    elif gender.casefold() == 'female':
        gender_numerical = 1

    # Create a feature vector from the user inputs
    features = [[gender_numerical, duration, heart_rate]]

    # Make the prediction using the loaded model
    predicted = calories_model.predict(features)
    pred_display = round(predicted[0], 2)

    # Returning the result from the function
    return(pred_display)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def generate():
    gender = str(request.form['gender'])
    duration = float(request.form['duration'])
    heart_rate = float(request.form['heart_rate'])

    prediction_result = predictions(gender, duration, heart_rate)

    return jsonify({'prediction_result': prediction_result})
    

if __name__ == '__main__':
    app.run(debug=True)
