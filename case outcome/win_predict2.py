import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
file_path = 'excel data.csv'
df = pd.read_csv(file_path)

# Selecting relevant features for training the model
features = ['Case Type', 'Evidence Volume', 'No. of Parties', 'Case Complexity']
target = 'Outcome'

# Extracting features and target variable
X = df[features].copy()  # Explicitly create a copy of the DataFrame
y = df[target]

# Remove whitespaces from 'Case Complexity' column
X['Case Complexity'] = X['Case Complexity'].str.strip()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Case Type', 'Evidence Volume', 'Case Complexity'])
    ],
    remainder='passthrough'
)

# Create the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Training the model
model.fit(X_train, y_train)

# Pickle the trained model for later use
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Function to make predictions on new data and get probability of winning
def predict_probability(new_data):
    # Load the trained model
    with open('model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Create a DataFrame from the received data
    new_data_df = pd.DataFrame([new_data], columns=features)

    # Remove whitespaces from 'Case Complexity' column
    new_data_df['Case Complexity'] = new_data_df['Case Complexity'].str.strip()

    # Make predictions on new data
    probability_of_winning = loaded_model.predict_proba(new_data_df)[:, 1]  # Probability of the positive class ('Won')

    return probability_of_winning[0]

# Example usage:
new_data_example = {
    'Case Type': 'Contractual Dispute',
    'Evidence Volume': 'High',
    'No. of Parties': 2,
    'Case Complexity': 'Medium'
}

probability_of_winning = predict_probability(new_data_example)
print(f'Probability of Winning: {probability_of_winning}')