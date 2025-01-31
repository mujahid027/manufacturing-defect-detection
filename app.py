# Import necessary libraries
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import joblib
import os

# Create a Flask application instance
app = Flask(__name__)

# Define global variables to store the dataset and model
dataset = None
model = None
scaler = StandardScaler()

# Define a route for the home page
@app.route('/')
def home():
    # Render the index.html template
    return render_template('index.html')

# Define a route for uploading a dataset
@app.route('/upload', methods=['POST'])
def upload_data():
    global dataset, scaler
    try:
        # Check if a file has been uploaded
        if 'file' not in request.files:
            # Return an error message if no file has been uploaded
            return jsonify({"error": "No file uploaded"}), 400
            
        # Get the uploaded file
        file = request.files['file']
        
        # Check if the file is a CSV file
        if not file.filename.endswith('.csv'):
            # Return an error message if the file is not a CSV file
            return jsonify({"error": "Only CSV files allowed"}), 400

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)
        
        # Define the required columns for the dataset
        required_columns = {'defectstatus'}  # Add other required columns
        
        # Check if the dataset contains all the required columns
        if not required_columns.issubset(df.columns):
            # Get the missing columns
            missing = required_columns - set(df.columns)
            # Return an error message with the missing columns
            return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400

        # Store the dataset in the global variable
        dataset = df
        
        # Reset the scaler for the new dataset
        scaler = StandardScaler()  
        
        # Render the train.html template
        return render_template('train.html')

    except Exception as e:
        # Return an error message with the exception details
        return jsonify({"error": str(e)}), 500

# Define a route for training a model
@app.route('/train', methods=['POST'])
def train_model():
    global dataset, model, scaler
    try:
        # Check if a dataset has been uploaded
        if dataset is None:
            # Return an error message if no dataset has been uploaded
            return jsonify({"error": "No dataset uploaded"}), 400

        # Prepare the data for training
        X = dataset.drop('defectstatus', axis=1)
        y = dataset['defectstatus']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Normalize the features using the StandardScaler
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Define the hyperparameter tuning space
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'max_iter': [500, 1000, 2000]
        }

        # Create a GridSearchCV instance for hyperparameter tuning
        model = GridSearchCV(
            LogisticRegression(),
            param_grid,
            cv=5,
            scoring='accuracy'
        )
        
        # Train the model using the training data
        model.fit(X_train_smote, y_train_smote)

        # Evaluate the model using the testing data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Save the trained model and scaler to files
        joblib.dump(model, 'trained_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')

        # Render the predict.html template with the training results
        return render_template('predict.html',
            message = "Model trained successfully",
            accuracy = round(accuracy, 2),
            f1_score= round(f1, 2))
            
    except Exception as e:
        # Return an error message with the exception details
        return jsonify({"error": str(e)}), 500

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a trained model exists
        if not os.path.exists('trained_model.joblib'):
            # Return an error message if no trained model exists
            return jsonify({"error": "Model not trained"}), 400

        # Load the trained model and scaler from files
        model = joblib.load('trained_model.joblib')
        scaler = joblib.load('scaler.joblib')   

        # Get the input data from the request
        csv_file = request.files['file']
        input_df = pd.read_csv(csv_file)

        # Preprocess the input data using the scaler
        scaled_input = scaler.transform(input_df)
        
        # Make predictions using the trained model
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[:, 1]

        # Render the results.html template with the prediction results
        return render_template('results.html', 
        predictions=prediction.tolist(), 
        probabilities=probability.tolist())
        
    except Exception as e:
        # Return an error message with the exception details
        return jsonify({"error": str(e)}), 400

# Run the Flask application
if __name__ == '__main__':
    # Create a directory for storing model files if it doesn't exist
    os.makedirs('models', exist_ok=True)
    # Run the application in debug mode
    app.run(debug=True)
