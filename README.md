# AI-Driven Fraud Detection, Job Matching, and Handbook Search API

This project is a Flask-based web application designed to showcase various AI-powered services, including:

- **Fraud Detection**: Predicting fraudulent transactions using machine learning (XGBoost) and oversampling techniques (SMOTE).
- **Job Matching**: Matching applicants to job listings based on text similarity using the `SentenceTransformer` model.
- **Handbook Search**: Providing AI-powered search across company handbooks to retrieve the most relevant policies based on user queries.

The app uses MongoDB to store various datasets, including applicants, jobs, transactions, and handbooks. The project includes endpoints to trigger fraud detection, job matching, and handbook policy retrieval.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Docker](#docker)
- [Future Improvements](#future-improvements)

## Technologies Used

- **Python**: Main programming language.
- **Flask**: Web framework used to build the API.
- **MongoDB**: Database used to store applicants, jobs, transactions, and handbooks.
- **XGBoost**: Machine learning algorithm used for fraud detection.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to handle class imbalance in fraud detection.
- **SentenceTransformer**: AI model used for semantic text similarity, aiding in job matching and handbook search.
- **pandas & numpy**: Data manipulation and numerical processing.
- **CORS (Cross-Origin Resource Sharing)**: Handled using `flask-cors` to allow requests from different domains.
- **dotenv**: For environment variable management.

## Features

- **Fraud Detection**: Detects fraudulent transactions by analyzing historical data and predicting new fraud cases. Utilizes the XGBoost machine learning model for predictions.
- **Job Matching**: Matches job applicants to job listings based on textual similarity using the `all-mpnet-base-v2` model from the `sentence-transformers` library.
- **Handbook Search**: Enables users to search company handbooks for relevant policies based on AI-powered semantic similarity analysis.

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- MongoDB instance (local or hosted)
- Access to the SentenceTransformer model (`all-mpnet-base-v2`)

### Step 1: Clone the Repository

```bash
git clone https://github.com/OldAlexhub/pyainexus.git
cd your-repo
```
