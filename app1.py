from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained pipeline
pipeline_path = 'trained_pipeline.pkl'
if not os.path.exists(pipeline_path):
    raise FileNotFoundError(f"🚫 Pipeline file not found: {pipeline_path}")
model = joblib.load(pipeline_path)

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

        # ✅ Directly use trained pipeline (includes preprocessing)
        df['Prediction'] = model.predict(df)

        return render_template("data.html", Y=df.to_html(classes='table', index=False))

    except Exception as e:
        return f"⚠️ Error while processing file: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
