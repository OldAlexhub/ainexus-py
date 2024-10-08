import spacy
from flask import Flask, jsonify, request, g
from flask_cors import CORS
import pymongo
import os
from dotenv import load_dotenv
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

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

# Load the spaCy model globally
nlp = spacy.load("en_core_web_sm")

# Home route
@app.route('/', methods=['GET'])
def home():
    return "Hello World!"

# Match applicant to job
@app.route('/match', methods=['GET'])
def match_applicant_to_job():
    try:
        db = get_db()
        applicants_collection = db['applicants']
        jobs_collection = db['jobs']

        # Fetch only the necessary fields
        applicant_data = pd.DataFrame(list(applicants_collection.find({}, {
            'Education': 1, 'Experience': 1, 'Skills': 1, 'Certifications': 1,
            'Employer1_Responsibilities': 1, 'Employer2_JobTitle': 1, 'Employer2_Responsibilities': 1,
            'Employer3_JobTitle': 1, 'Employer3_Responsibilities': 1, 'ApplicantName': 1
        })))
        
        jobs_data = pd.DataFrame(list(jobs_collection.find({}, {
            'JobTitle': 1, 'Description': 1, 'Responsibilities': 1, 'Requirements': 1
        })))

        # Flatten lists into strings if needed
        def flatten_list_to_string(value):
            if isinstance(value, list):
                return ' '.join(value)
            return str(value)

        # Create text representations
        application_text = ' '.join(flatten_list_to_string(applicant_data[col]) for col in applicant_data if col in applicant_data)
        job_text = ' '.join(flatten_list_to_string(jobs_data[col].iloc[0]) for col in jobs_data)

        # Process text with spaCy and compute similarity
        applicant_doc = nlp(application_text)
        job_doc = nlp(job_text)
        similarity = applicant_doc.similarity(job_doc)
        match_percentage = round(similarity * 100, 2)

        # Prepare results
        name = applicant_data['ApplicantName'].iloc[0]
        title = jobs_data['JobTitle'].iloc[0]
        results = {
            'name': name,
            'title': title,
            'match': match_percentage,
        }

        return jsonify([results]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Handbook search (optimized)
@app.route('/handbook/<searchedTerm>', methods=['GET'])
def get_answers(searchedTerm):
    try:
        db = get_db()
        handbook_collection = db['handbooks']

        # Fetch only required fields for matching
        handbook_entries = list(handbook_collection.find({}, {'_id': 0, 'PolicyTitle': 1, 'Policy': 1}))

        # Process the searched term using spaCy
        query_doc = nlp(searchedTerm)

        # Iterate over handbook policies and compute similarity
        policies = [entry['Policy'] for entry in handbook_entries]
        similarities = []
        for policy in policies:
            policy_doc = nlp(policy)
            similarity = query_doc.similarity(policy_doc)
            similarities.append(similarity)

        # Find the best match based on highest similarity score
        top_match_index = similarities.index(max(similarities))
        best_match = handbook_entries[top_match_index]

        result = {
            "searchedTerm": searchedTerm,
            "bestMatchPolicyTitle": best_match['PolicyTitle'],
            "bestMatchPolicy": best_match['Policy'],
            "confidenceScore": round(similarities[top_match_index], 4)  # Rounded for cleaner display
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Fraud detection
@app.route('/fraud', methods=['GET'])
def fraud_decision():
    try:
        db = get_db()
        fraud_collection = db['frauds']
        transactions_collection = db['transactions']

        # Fetch labeled fraud data and transaction data
        fraud_data = pd.DataFrame(list(fraud_collection.find({}, {'_id': 0})))
        transaction_data = pd.DataFrame(list(transactions_collection.find({}, {'_id': 0})))

        # Ensure 'Is_Fraudulent' column exists
        if 'Is_Fraudulent' not in fraud_data.columns:
            return jsonify({'error': 'Is_Fraudulent column not found'}), 400

        # Define X and y for training
        X = fraud_data.drop(columns=['Is_Fraudulent'])
        y = fraud_data['Is_Fraudulent']

        # Use get_dummies for categorical encoding
        X = pd.get_dummies(X, drop_first=True)
        transaction_data_encoded = pd.get_dummies(transaction_data, drop_first=True)
        transaction_data_encoded = transaction_data_encoded.reindex(columns=X.columns, fill_value=0)

        # Use SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)

        # Make predictions and filter fraudulent transactions
        transaction_predictions = xgb_model.predict(transaction_data_encoded)
        transaction_data['predicted_fraud'] = transaction_predictions
        fraudulent_transactions = transaction_data[transaction_data['predicted_fraud'] == 1]

        return jsonify(fraudulent_transactions.to_dict(orient='records')), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
