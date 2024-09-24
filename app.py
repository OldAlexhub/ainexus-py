from flask import Flask, jsonify, request, g
from flask_cors import CORS
import numpy as np
import pandas as pd
import pymongo
import os
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # CORS should be configured to allow specific origins in production

# MongoDB connection (Fork-Safe)
def get_db():
    if 'db' not in g:
        mongo_url = os.getenv('MONGO_URL')
        client = pymongo.MongoClient(mongo_url)
        g.db = client['test']  # Use your actual DB
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.client.close()

# Load a more efficient AI model (smaller version)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Home route
@app.route('/', methods=['GET'])
def home():
    return "Hello World!"

# Match applicant to job
@app.route('/match', methods=['GET'])
def match_applicant_to_job():
    db = get_db()
    applicants_collection = db['applicants']
    jobs_collection = db['jobs']

    # Fetch the data from MongoDB collections
    applicant_data = pd.DataFrame(list(applicants_collection.find()))
    jobs_data = pd.DataFrame(list(jobs_collection.find()))

    # Function to flatten lists into strings
    def flatten_list_to_string(value):
        if isinstance(value, list):
            return ' '.join(value)
        return str(value)

    # Create text representations of the applicant and job
    application_text = ' '.join(flatten_list_to_string(applicant_data[col]) for col in [
        'Education', 'Experience', 'Skills', 'Certifications', 'Employer1_Responsibilities',
        'Employer2_JobTitle', 'Employer2_Responsibilities', 'Employer3_JobTitle', 'Employer3_Responsibilities'] 
        if col in applicant_data)

    job_text = ' '.join(flatten_list_to_string(jobs_data[col].iloc[0]) for col in [
        'JobTitle', 'Description', 'Responsibilities', 'Requirements'] 
        if col in jobs_data)

    # Calculate similarity between applicant and job descriptions
    embeddings = model.encode([application_text, job_text])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    match_percentage = round(similarity * 100, 2)

    # Prepare the results DataFrame
    name = applicant_data['ApplicantName'].iloc[0]
    title = jobs_data['JobTitle'].iloc[0]
    results = pd.DataFrame({
        'name': [name],
        'title': [title],
        'match': [match_percentage],
    })

    # Convert DataFrame to dictionary and return as JSON
    return jsonify(results.to_dict(orient='records'))

# Handbook search
@app.route('/handbook/<searchedTerm>', methods=['GET'])
def get_answers(searchedTerm):
    db = get_db()
    handbook_collection = db['handbooks']
    
    # Get all the handbook entries
    handbook_entries = list(handbook_collection.find({}, {'_id': 0, 'PolicyTitle': 1, 'Policy': 1}))

    # Create a list of handbook policy texts
    policies = [entry['Policy'] for entry in handbook_entries]

    # Encode the searched term and the policies using the AI model
    query_embedding = model.encode(searchedTerm)
    policy_embeddings = model.encode(policies)

    # Compute similarity scores
    similarities = util.cos_sim(query_embedding, policy_embeddings).numpy()[0]

    # Get the best match (or top matches)
    top_match_index = similarities.argmax()
    best_match = handbook_entries[top_match_index]
    best_score = similarities[top_match_index]

    # Format the result
    result = {
        "searchedTerm": searchedTerm,
        "bestMatchPolicyTitle": best_match['PolicyTitle'],
        "bestMatchPolicy": best_match['Policy'],
        "confidenceScore": float(best_score)
    }

    return jsonify(result)

# Fraud detection
@app.route('/fraud', methods=['GET'])
def fraud_decision():
    db = get_db()
    fraud_collection = db['frauds']
    transactions_collection = db['transactions']

    print('fraud has been triggered')

    # Fetch the labeled fraud data and unlabeled transaction data
    fraud_data = pd.DataFrame(list(fraud_collection.find()))
    transaction_data = pd.DataFrame(list(transactions_collection.find()))
    
    # Drop the '_id' column from both datasets
    fraud_data = fraud_data.drop(columns=['_id'], errors='ignore')
    transaction_data = transaction_data.drop(columns=['_id'], errors='ignore')

    # Ensure 'Is_Fraudulent' is the target variable (if not present, return error)
    if 'Is_Fraudulent' not in fraud_data.columns:
        return jsonify({'error': 'Is_Fraudulent column not found in the dataset'}), 400

    # Define X and y for training
    X = fraud_data.drop(columns=['Is_Fraudulent'])
    y = fraud_data['Is_Fraudulent']

    # Convert categorical columns to dummy variables for both datasets
    X = pd.get_dummies(X, drop_first=True)
    transaction_data_encoded = pd.get_dummies(transaction_data, drop_first=True)

    # Ensure the transaction data has the same columns as the training data
    transaction_data_encoded = transaction_data_encoded.reindex(columns=X.columns, fill_value=0)

    # Oversample with SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train an XGBoost classifier
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Make predictions on the new transaction data
    transaction_predictions = xgb_model.predict(transaction_data_encoded)
    transaction_data['predicted_fraud'] = transaction_predictions

    # Filter the fraudulent transactions
    fraudulent_transactions = transaction_data[transaction_data['predicted_fraud'] == 1]

    # Return the fraudulent transactions as JSON
    return jsonify(fraudulent_transactions.to_dict(orient='records'))

if __name__ == '__main__':
    # Set debug mode based on the environment variable
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Run Flask app with explicit host and port
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
