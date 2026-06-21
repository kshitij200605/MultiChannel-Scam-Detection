import pandas as pd
import numpy as np
import os
import uuid
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import wordnet

# Download wordnet quietly if not already present
nltk.download('wordnet', quiet=True)

# Define paths relative to this script
RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw'))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))

def load_and_merge_datasets():
    print("1. Loading and merging datasets...")
    # NOTE: These are placeholder file paths and logic.
    # Replace with actual files once they are downloaded into data/raw/
    datasets = []
    
    # 1. SMS Spam Collection
    sms_path = os.path.join(RAW_DATA_DIR, "sms_spam.csv")
    if os.path.exists(sms_path):
        sms_df = pd.read_csv(sms_path)
        sms_df = sms_df.rename(columns={'text_col': 'text', 'label_col': 'original_label'})
        sms_df['source_dataset'] = 'sms_spam_collection'
        datasets.append(sms_df)
    
    # 2. Enron Email Dataset
    enron_path = os.path.join(RAW_DATA_DIR, "enron.csv")
    if os.path.exists(enron_path):
        enron_df = pd.read_csv(enron_path)
        enron_df = enron_df.rename(columns={'message': 'text', 'type': 'original_label'})
        enron_df['source_dataset'] = 'enron_email'
        datasets.append(enron_df)

    # 3. Nazario Phishing Corpus
    nazario_path = os.path.join(RAW_DATA_DIR, "nazario.csv")
    if os.path.exists(nazario_path):
        nazario_df = pd.read_csv(nazario_path)
        nazario_df = nazario_df.rename(columns={'body': 'text', 'label': 'original_label'})
        nazario_df['source_dataset'] = 'nazario_phishing'
        datasets.append(nazario_df)

    # 4. Fraudulent Job Postings Dataset
    job_path = os.path.join(RAW_DATA_DIR, "fake_job_postings.csv")
    if os.path.exists(job_path):
        job_df = pd.read_csv(job_path)
        # Often this dataset uses 'description' for text and 'fraudulent' (0 or 1) for label
        job_df['text'] = job_df['title'].fillna('') + " " + job_df['description'].fillna('')
        job_df['original_label'] = job_df['fraudulent'].apply(lambda x: 'fraudulent' if x == 1 else 'legitimate')
        job_df = job_df[['text', 'original_label']]
        job_df['source_dataset'] = 'fraudulent_job_postings'
        datasets.append(job_df)

    # 5. Phishing URL Dataset
    phish_url_path = os.path.join(RAW_DATA_DIR, "phishing_urls.csv")
    if os.path.exists(phish_url_path):
        phish_url_df = pd.read_csv(phish_url_path)
        phish_url_df = phish_url_df.rename(columns={'url': 'text', 'status': 'original_label'})
        phish_url_df['source_dataset'] = 'phishing_url_dataset'
        datasets.append(phish_url_df)

    # 6. OpenPhish
    openphish_path = os.path.join(RAW_DATA_DIR, "openphish.csv")
    if os.path.exists(openphish_path):
        op_df = pd.read_csv(openphish_path)
        # OpenPhish is usually a list of malicious URLs without a label column
        op_df = op_df.rename(columns={'url': 'text'})
        if 'text' not in op_df.columns and len(op_df.columns) == 1:
            op_df = op_df.rename(columns={op_df.columns[0]: 'text'})
        op_df['original_label'] = 'phishing'
        op_df['source_dataset'] = 'openphish'
        datasets.append(op_df)

    # 7. Custom collected scam samples
    custom_path = os.path.join(RAW_DATA_DIR, "custom_scam_samples.csv")
    if os.path.exists(custom_path):
        custom_df = pd.read_csv(custom_path)
        custom_df = custom_df.rename(columns={'content': 'text', 'category': 'original_label'})
        custom_df['source_dataset'] = 'custom_collected'
        datasets.append(custom_df)

    # If no real data is placed in the folders yet, use dummy data to test the pipeline
    if not datasets:
        print("   -> Warning: No raw datasets found. Using a dummy dataset for demonstration.")
        datasets.append(pd.DataFrame({
            'text': [
                'Win a free iPhone now! Click here.', 
                'Hi Mom, can you call me when you get this?', 
                'Urgent: Your bank account is locked due to suspicious activity.', 
                'Meeting at 5pm tomorrow.',
                'Your OTP is 123456. Do not share this with anyone.',
                'Invest in Crypto now and get 500% returns guaranteed!'
            ],
            'original_label': ['spam', 'ham', 'phishing', 'normal', 'otp_fraud', 'crypto_scam'],
            'source_dataset': ['dummy_data'] * 6
        }))
        
    unified_df = pd.concat(datasets, ignore_index=True)
    return unified_df

def standardize_labels(df):
    print("2. Standardizing labels...")
    # Map original dataset labels to the 10 final classes
    label_map = {
        'ham': 'legitimate',
        'normal': 'legitimate',
        'legit': 'legitimate',
        'spam': 'phishing',
        'phishing': 'phishing',
        'fraudulent': 'fake_job',
        'otp_fraud': 'otp_fraud',
        'investment': 'investment_scam',
        'crypto_scam': 'crypto_scam',
        'impersonation': 'impersonation',
        'lottery': 'lottery_scam',
        'tech_support': 'tech_support_scam',
        'upi_fraud': 'upi_fraud'
    }
    
    # Apply mapping. If an unknown label is encountered, default it to 'phishing' or keep it.
    df['final_label'] = df['original_label'].map(label_map).fillna('legitimate')
    return df

def split_dataset(df):
    print("3. Train/Validation/Test Split (70/15/15)...")
    # Safely handle case where class counts are too small for stratification in dummy data
    stratify_col = df['final_label'] if len(df) > 10 else None
    
    try:
        # First split: 70% train, 30% temp
        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=stratify_col, random_state=42
        )
        # Second split: split the 30% temp equally into validation (15%) and test (15%)
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df['final_label'] if stratify_col is not None else None, random_state=42
        )
    except ValueError:
        # Fallback for very small dummy datasets
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
        
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['split_group'] = 'train'
    val_df['split_group'] = 'val'
    test_df['split_group'] = 'test'
    
    return pd.concat([train_df, val_df, test_df], ignore_index=True)

def balance_classes_smote():
    # Note: SMOTE operates on numeric features. 
    # This will be applied later in the ML pipeline (Phase 1) after text vectorization (e.g., TF-IDF).
    print("4. Class Balancing Strategy...")
    print("   -> SMOTE will be applied to TF-IDF vectors during the Model Training phase.")

def augment_text(text):
    # Data Augmentation Strategy: Synonym Replacement
    words = text.split()
    augmented_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms and len(synonyms[0].lemmas()) > 1:
            # Replace with a synonym
            synonym = synonyms[0].lemmas()[1].name()
            augmented_words.append(synonym.replace('_', ' '))
        else:
            augmented_words.append(word)
    return " ".join(augmented_words)

def apply_data_augmentation(df):
    print("5. Data Augmentation Strategy...")
    # Only augment the training set to prevent data leakage
    train_df = df[df['split_group'] == 'train']
    
    class_counts = train_df['final_label'].value_counts()
    median_count = class_counts.median() if len(class_counts) > 0 else 0
    # Identify minority classes
    minority_classes = class_counts[class_counts <= median_count].index.tolist()
    
    augmented_rows = []
    for index, row in train_df.iterrows():
        if row['final_label'] in minority_classes:
            # Create an augmented version of the text
            aug_text = augment_text(row['text'])
            # Don't add identical texts
            if aug_text != row['text']:
                new_row = row.copy()
                new_row['text'] = aug_text
                new_row['original_label'] = row['original_label'] + "_augmented"
                augmented_rows.append(new_row)
            
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, aug_df], ignore_index=True)
        print(f"   -> Generated {len(augmented_rows)} augmented samples for minority classes.")
    return df

def generate_schema(df):
    print("6. Formatting final CSV Schema...")
    # Add UUIDs if missing
    if 'id' not in df.columns:
        df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
    # Enforce final column structure
    final_columns = ['id', 'text', 'source_dataset', 'original_label', 'final_label', 'split_group']
    df = df[final_columns]
    
    output_path = os.path.join(PROCESSED_DATA_DIR, 'unified_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"   -> Unified Dataset successfully saved to {output_path}")

if __name__ == "__main__":
    # Ensure data directories exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Execute Data Pipeline
    df = load_and_merge_datasets()
    df = standardize_labels(df)
    df = split_dataset(df)
    balance_classes_smote()
    df = apply_data_augmentation(df)
    generate_schema(df)
