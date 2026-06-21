import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FILE = os.path.join(BASE_DIR, 'data/processed/unified_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models/baseline')
RESULTS_DIR = os.path.join(BASE_DIR, 'models/results')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Processed dataset not found at {DATA_FILE}. Please run dataset_pipeline.py first.")
    df = pd.read_csv(DATA_FILE)
    # Ensure text is string and drop NaNs
    df = df.dropna(subset=['text', 'final_label'])
    df['text'] = df['text'].astype(str)
    return df

def extract_features(df):
    print("Vectorizing text using TF-IDF...")
    train_df = df[df['split_group'] == 'train']
    val_df = df[df['split_group'] == 'val']
    test_df = df[df['split_group'] == 'test']

    # Using max_features to keep the matrix manageable, along with unigrams and bigrams
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    X_test = vectorizer.transform(test_df['text'])

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['final_label'])
    y_val = le.transform(val_df['final_label'])
    y_test = le.transform(test_df['final_label'])

    print(f"Applying SMOTE to balance {X_train.shape[0]} training samples...")
    # Determine n_neighbors dynamically if sample size is extremely small (e.g. for dummy dataset)
    min_samples = pd.Series(y_train).value_counts().min()
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

    if k_neighbors > 0 and len(np.unique(y_train)) > 1:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        print("Not enough samples for SMOTE (likely running on very small dummy data). Using original distribution.")
        X_train_res, y_train_res = X_train, y_train

    print(f"Training samples after SMOTE: {X_train_res.shape[0]}")

    # Save the vectorizer and label encoder for future inference
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    return (X_train_res, y_train_res), (X_val, y_val), (X_test, y_test), le.classes_

def evaluate_model(name, model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    print(f"\n--- {name} Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Generate Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{name.replace(" ", "_")}_confusion_matrix.png'))
    plt.close()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def main():
    print("=== Phase 1: Training Baseline Models ===")
    df = load_data()
    train_data, val_data, test_data, class_names = extract_features(df)
    
    X_train, y_train = train_data
    X_test, y_test = test_data

    results = {}

    # 1. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
    results['Random_Forest'] = evaluate_model("Random Forest", rf, X_test, y_test, class_names)

    # 2. XGBoost
    print("\nTraining XGBoost...")
    # Map the labels purely to consecutive integers starting from 0 (handled by LabelEncoder already)
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    results['XGBoost'] = evaluate_model("XGBoost", xgb, X_test, y_test, class_names)

    # Store results JSON
    with open(os.path.join(RESULTS_DIR, 'baseline_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAll models trained and saved to {MODEL_DIR}")
    print(f"Confusion matrices and metrics saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
