import joblib
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load trained logistic regression model
model = joblib.load('logistic_model.pkl')

# HTML template with embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Assessment Predictor</title>
    <style>
        :root {
            --primary: #2563eb;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --border: #cbd5e1;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 40px 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        h2 { text-align: center; color: var(--primary); margin-bottom: 24px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 6px;
            text-transform: capitalize;
        }
        input, select {
            padding: 10px;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 0.95rem;
        }
        button {
            grid-column: 1 / -1;
            margin-top: 16px;
            padding: 12px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background-color: #1d4ed8; }
        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            text-align: center;
        }
        .result-title { font-size: 1.1rem; font-weight: bold; }
        .result-val { font-size: 1.5rem; color: var(--primary); font-weight: 800; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Health Risk Level Prediction</h2>
        <form action="/predict" method="POST" class="grid">
            <div class="form-group">
                <label>Age</label>
                <input type="number" name="age" required value="45">
            </div>
            <div class="form-group">
                <label>Gender</label>
                <input type="text" name="gender" placeholder="e.g., Male/Female" required>
            </div>
            <div class="form-group">
                <label>City</label>
                <input type="text" name="city" placeholder="City name" required>
            </div>
            <div class="form-group">
                <label>BMI</label>
                <input type="number" step="0.1" name="bmi" required value="24.5">
            </div>
            <div class="form-group">
                <label>Family History Diabetes</label>
                <input type="text" name="family_history_diabetes" placeholder="Yes/No" required>
            </div>
            <div class="form-group">
                <label>Physical Activity</label>
                <input type="text" name="physical_activity_level" placeholder="Low/Moderate/High" required>
            </div>
            <div class="form-group">
                <label>Diet Type</label>
                <input type="text" name="diet_type" placeholder="Balanced/Keto/etc." required>
            </div>
            <div class="form-group">
                <label>Smoking Status</label>
                <input type="text" name="smoking_status" placeholder="Never/Former/Current" required>
            </div>
            <div class="form-group">
                <label>Alcohol Consumption</label>
                <input type="text" name="alcohol_consumption" placeholder="None/Moderate/High" required>
            </div>
            <div class="form-group">
                <label>Hours Sleep / Night</label>
                <input type="number" step="0.1" name="hours_sleep_per_night" required value="7">
            </div>
            <div class="form-group">
                <label>Stress Level</label>
                <input type="number" name="stress_level" min="1" max="10" required value="5">
            </div>
            <div class="form-group">
                <label>Fasting Blood Sugar</label>
                <input type="number" step="0.1" name="fasting_blood_sugar" required value="95">
            </div>
            <div class="form-group">
                <label>HbA1c Level</label>
                <input type="number" step="0.1" name="hba1c_level" required value="5.6">
            </div>
            <div class="form-group">
                <label>BP Systolic</label>
                <input type="number" name="blood_pressure_systolic" required value="120">
            </div>
            <div class="form-group">
                <label>BP Diastolic</label>
                <input type="number" name="blood_pressure_diastolic" required value="80">
            </div>
            <div class="form-group">
                <label>Waist Circumference (cm)</label>
                <input type="number" step="0.1" name="waist_circumference_cm" required value="85">
            </div>
            <div class="form-group">
                <label>Income Bracket</label>
                <input type="text" name="income_bracket" placeholder="Low/Medium/High" required>
            </div>
            <button type="submit">Predict Risk Category</button>
        </form>

        {% if prediction %}
        <div class="result-box">
            <span class="result-title">Predicted Risk Level:</span>
            <div class="result-val">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    
    # Map numerical inputs to numeric types
    numeric_fields = [
        'age', 'bmi', 'hours_sleep_per_night', 'stress_level', 
        'fasting_blood_sugar', 'hba1c_level', 'blood_pressure_systolic', 
        'blood_pressure_diastolic', 'waist_circumference_cm'
    ]
    for key in numeric_fields:
        if key in form_data:
            form_data[key] = float(form_data[key])
            
    # Convert form dict to single-row DataFrame matching trained columns
    input_df = pd.DataFrame([form_data])
    
    # Predict risk class
    prediction = model.predict(input_df)[0]
    
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
