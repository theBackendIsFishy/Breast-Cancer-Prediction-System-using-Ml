import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/final")
def final():
    return "<h1>Hello all good</h1>"


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get values from form
#         perimeter_mean       = float(request.form['perimeter_mean'])
#         concave_points_mean  = float(request.form['concave points_mean'])
#         radius_worst         = float(request.form['radius_worst'])
#         perimeter_worst      = float(request.form['perimeter_worst'])
#         concave_points_worst = float(request.form['concave points_worst'])

#         # Arrange features (same order as training)
#         features = np.array([[perimeter_mean,
#                               concave_points_mean,
#                               radius_worst,
#                               perimeter_worst,
#                               concave_points_worst]])

#         # Apply scaling
#         features = scaler.transform(features)

#         # Prediction
#         prediction = model.predict(features)[0]

#         #  Updated logic (based on your mapping)
#         if prediction == 1:
#             result = "Malignant (Cancer Detected)"
#         else:
#             result = "Benign (No Cancer)"

#         return render_template('result.html', prediction=result)

#     except Exception as e:
#         return f"Error: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        perimeter_mean       = float(request.form['perimeter_mean'])
        concave_points_mean  = float(request.form['concave points_mean'])
        radius_worst         = float(request.form['radius_worst'])
        perimeter_worst      = float(request.form['perimeter_worst'])
        concave_points_worst = float(request.form['concave points_worst'])

        features = np.array([[perimeter_mean,
                              concave_points_mean,
                              radius_worst,
                              perimeter_worst,
                              concave_points_worst]])

        features = scaler.transform(features)

        prediction = model.predict(features)[0]

        #  ADD THESE 3 LINES
        proba         = model.predict_proba(features)[0]
        benign_pct    = round(proba[0] * 100, 1)
        malignant_pct = round(proba[1] * 100, 1)
        confidence    = round(proba[prediction] * 100, 1)

        if prediction == 1:
            result = "Malignant (Cancer Detected)"
        else:
            result = "Benign (No Cancer)"

        #  PASS THEM TO TEMPLATE
        return render_template('result.html',
                               prediction=result,
                               confidence=confidence,
                               benign_pct=benign_pct,
                               malignant_pct=malignant_pct)

    except Exception as e:
        return f"Error: {str(e)}"
if __name__ == "__main__":
    app.run(debug=True ) #ssl_context="adhoc"
