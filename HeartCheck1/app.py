# Dependencies used to build the web app
from flask import Flask, render_template, jsonify, request
from werkzeug.exceptions import HTTPException
import traceback
import io

# Used to load and run the model
# Packages: scikit-learn, joblib
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
model = None


def load_model():
    """
    This method loads our model, so that we can use it to perform predictions.
    """

    global model
    if not model:
        # print("--->>>> loading model...")
        # TODO: Change the filename to match your model's filename
        model = joblib.load("heart_classifier.pkl")
    return model


# Homepage: The heart health form
@app.route('/')
def form():
    # Get the values if specified in the URL (so we can edit them)
    values = request.values

    return render_template('form.html', form_values=values)


@app.route('/process_form', methods=["POST"])
def process_form():
    # Get the values that were submitted in the form, and
    # convert them to correct numeric format (integer or floats)
    values = {
        'age': int(request.form['age']),
        'bp_systolic': int(request.form['bp_systolic']),
        'bp_diastolic': int(request.form['bp_diastolic']),
        'weight_kg': float(request.form['weight_kg']),
        'height_cm': float(request.form['height_cm']),
        'cholesterol': int(request.form['cholesterol']),
        'pa': int(request.form['pa']),
        'sm': int(request.form['sm']),
        'hm': int(request.form['hm']),
        
    }

    # We do not have a BMI field on our form, but our model requires it.
    # We can calculate BMI using height and weight.
    bmi = calculate_bmi(values['height_cm'], values['weight_kg'])

    # This is a dictionary of the values that our model expects, as well
    # as a human-readable description of what each value means, so we can
    # display this to the user.
    # 0=Normal, 1=Above Normal, 2=Well Above Normal
    cholesterol_descriptions = {
        0: "Normal",
        1: "Above Normal",
        2: "Well Above Normal",
    }
pa_descriptions = {
        0: "Yes",
        1: "No",
       
    }
sm_descriptions = {
        0: "Never",
        1: "Occasional",
        2:"Regular",
       
    }
hm_descriptions = {
        0: "Yo",
        1: "Yes",
       
    }
    # These are the values that we will display on the results page
    input_values = {
        "Age": values['age'],
        "Blood Pressure": "%s/%s" % (values['bp_systolic'], values['bp_diastolic']),
        "Weight": "%s kg" % values['weight_kg'],
        "Height": "%s cm" % values['height_cm'],
        "BMI": bmi,
        "Cholesterol": cholesterol_descriptions[values['cholesterol']],
       "Physically active?": pa_descriptions[values['pa']],
        "Smoking?": sm_descriptions[values['sm']],
        "Familial Hypertrophic Cardiomyopathy(Hereditary heart problems)": hm_descriptions[values['hm']]
    }

    # Load the model & model params
    model = load_model()
    model_params = [[
        values['bp_systolic'],
        values['bp_diastolic'],
        values['age'],
        values['cholesterol'],
        bmi
    ]]

    # Use our model to perform predictions

    # model.predict returns an array containing the prediction
    #    e.g. => [[0]]
    prediction = model.predict(model_params)[0]

    # model.predict_proba returns an array containing the probabilities of each class
    #    e.g. => [[0.65566831, 0.34433169]]
    probabilities = model.predict_proba(model_params)[0]

    return render_template('results.html', prediction=prediction, probabilities=probabilities, input_values=input_values, form_values=values)


def calculate_bmi(height_cm, weight_kg):
    """
    Calculates BMI given height (in kg), and weight (in cm)
    BMI Formula: kg / m^2
    Output is BMI, rounded to one decimal digit
    """

    # Input height is in cm, so we divide by 100 to convert to metres
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


# Start the server
if __name__ == "__main__":
    print("* Starting Flask server..."
          "please wait until server has fully started")
    # debug=True options allows us to view our changes without restarting the server.
    app.run(host='0.0.0.0', debug=True)
