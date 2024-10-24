from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load the pre-trained model
model = load('HEART_DISEASE_MODEL.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    conclusions = []
    
    if request.method == 'POST':
        # Get form data
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
        if prediction == 1:
            result = "Abnormality Detected"
            conclusions.append("As per input data, heart abnormality detected. Consult your doctor immediately.")
        else:
            result = "No Abnormality Detected"
            conclusions.append("As per input data, no heart disease detected. Maintain a healthy lifestyle.")
        
        # Add additional conclusions based on input
        if age > 50:
            conclusions.append("Your age is above 50, which may increase the risk of heart disease. Ensure regular check-ups and maintain a healthy lifestyle to mitigate risks.")
        
        if chol > 200:
            conclusions.append("Your cholesterol level is higher than the normal range (200 mg/dl). To reduce cholesterol, consider a low-cholesterol diet, regular exercise, and medication if prescribed by your doctor.")
        
        if fbs == '1':
            conclusions.append("Your fasting blood sugar level is higher than normal. Control sugar intake, exercise regularly, and consult your doctor for further evaluation.")

    # Render the index.html with the result and conclusions
    return render_template('index.html', result=result, conclusions=conclusions)

if __name__ == '__main__':
    app.run(debug=True)
