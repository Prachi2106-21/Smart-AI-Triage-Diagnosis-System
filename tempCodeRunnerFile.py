import os
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# Directories & Files
# ----------------------------
MODEL_DIR = 'analysis/artifacts'
os.makedirs(MODEL_DIR, exist_ok=True)

LABELS_FILE = 'data/labels4.csv'
REPORTS_FOLDER = 'data/reports'
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# ----------------------------
# 1️⃣ Unsupervised training
# ----------------------------
def train_unsupervised(df_features):
    print("\n[Unsupervised] Training IsolationForest...")
    X = df_features.fillna(df_features.median())
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)

    dump({'iso': iso, 'columns': list(X.columns)}, os.path.join(MODEL_DIR, 'iso.joblib'))
    print("✅ IsolationForest saved to artifacts/iso.joblib")

# ----------------------------
# 2️⃣ Supervised training
# ----------------------------
def train_supervised(df_labels):
    print("\n[Supervised] Training RandomForest...")

    if 'label' not in df_labels.columns:
        raise ValueError("labels4.csv must contain a 'label' column")

    # Features & target
    X = df_labels.drop(columns=['label', 'abnormalities'], errors='ignore')
    y = df_labels['label']

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"🔍 Accuracy: {acc:.2f}")
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model + imputer + columns
    dump({
        'model': rf,
        'imputer': imputer,
        'columns': list(X.columns)
    }, os.path.join(MODEL_DIR, 'rf.joblib'))
    print("✅ RandomForest saved to artifacts/rf.joblib")

# ----------------------------
# 3️⃣ Load data & train models
# ----------------------------
if __name__ == '__main__':
    # Load labels4.csv
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"{LABELS_FILE} not found. Generate it first from predictions.csv.")

    df_labels = pd.read_csv(LABELS_FILE)
    df_features = df_labels.drop(columns=['label', 'abnormalities'], errors='ignore')

    # Train models
    train_unsupervised(df_features)
    train_supervised(df_labels)

    print("\n✅ All models trained & evaluated successfully!")
