from flask import Flask, jsonify, request, render_template
from joblib import load

app = Flask(__name__)

# Load the pre-trained model
model = load('HEART_DISEASE_MODEL.joblib')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Serve the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from request
    # name = request.form['name']
    age = int(request.form['age'])
    sex = request.form['sex']
    chest_pain = request.form['chest_pain']
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = request.form['fbs']
    rest_ecg = request.form['rest_ecg']
    thalach = int(request.form['thalach'])
    ex_ang = request.form['ex_ang']
    old_peak = float(request.form['old_peak'])
    slope = request.form['slope']
    ca = int(request.form['ca'])
    thal = request.form['thal']
    
    # Prepare the input data for the model
    inputs = [age, sex, chest_pain, trestbps, chol, fbs, rest_ecg, thalach, ex_ang, old_peak, slope, ca, thal]
    
    # Make prediction
    prediction = model.predict([inputs])[0]
    
    # Determine result
    result = "Abnormality Detected" if prediction == 1 else "No Abnormality Detected"
    
    # Prepare conclusions
    conclusions = []
    if age > 50:
        conclusions.append("Your age is above 50, which may increase the risk of heart disease.")
    
    if chol > 200:
        conclusions.append("Your cholesterol level is higher than normal (200 mg/dl). Consider lifestyle changes.")
    
    if fbs == '1':
        conclusions.append("Your fasting blood sugar level is higher than normal. Control sugar intake.")

    return jsonify({
        'result': result,
        'conclusions': conclusions
    })

if __name__ == '__main__':
    app.run(debug=True)
