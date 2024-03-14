from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Features used for prediction
features = ['Case Type', 'Evidence Volume', 'No. of Parties', 'Case Complexity']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        new_data = {
            'Case Type': request.form['case_type'],
            'Evidence Volume': request.form['evidence_volume'],
            'No. of Parties': int(request.form['num_parties']),
            'Case Complexity': request.form['case_complexity']
        }

        # Create a DataFrame from the received data
        new_data_df = pd.DataFrame([new_data], columns=features)

        # Remove whitespaces from 'Case Complexity' column
        new_data_df['Case Complexity'] = new_data_df['Case Complexity'].str.strip()

        # Make predictions on new data
        probability_of_winning = loaded_model.predict_proba(new_data_df)[:, 1]

        return render_template('result.html', probability_of_winning=probability_of_winning[0])

if __name__ == '__main__':
    app.run(debug=True)

