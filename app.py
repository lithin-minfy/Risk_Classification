from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model pipeline
model_path = 'model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚫 Model file not found: {model_path}")
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods=['POST'])
def success():
    if 'file' not in request.files:
        return "❌ No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "❌ No file selected."

    try:
        # Load uploaded CSV
        df = pd.read_csv(file)

        # ✅ Match training-time preprocessing

        # Drop unused columns
        df.drop(['ID', 'ZIP Code', 'Experience'], axis=1, inplace=True)

        # Log-transform skewed features
        for col in ['Income', 'CCAvg', 'Mortgage']:
            df[col] = np.log1p(df[col])  # log(1 + x)

        # Add HasMortgage feature
        df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)

        # Drop Mortgage as done during training
        df.drop('Mortgage', axis=1, inplace=True)

        # Make predictions
        df['Prediction'] = model.predict(df)

        # Render the predictions in an HTML table
        return render_template("data.html", Y=df.to_html(classes='table', index=False))

    except Exception as e:
        return f"⚠️ Error while processing file: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
