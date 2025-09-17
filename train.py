import os
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Multi-label
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# ----------------------------
# Directories & Files
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'analysis', 'artifacts')
os.makedirs(MODEL_DIR, exist_ok=True)

LABELS_FILE = os.path.join(BASE_DIR, 'data', 'labels4.csv')
REPORTS_FOLDER = os.path.join(BASE_DIR, 'data', 'reports')
os.makedirs(REPORTS_FOLDER, exist_ok=True)

DIAGNOSTIC_CSV = os.path.join(BASE_DIR, 'data', 'diagnostic_dataset_with_symptoms_meds.csv')

# ----------------------------
# 1️⃣ Unsupervised training
# ----------------------------
def train_unsupervised(df_features):
    print("\n[Unsupervised] Training IsolationForest...")
    X = df_features.fillna(df_features.median())
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    # Save model + columns
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

    # Save model + imputer + columns ✅
    dump({
        'model': rf,
        'imputer': imputer,
        'columns': list(X.columns)
    }, os.path.join(MODEL_DIR, 'rf.joblib'))
    
    print("✅ RandomForest saved to artifacts/rf.joblib")

# ----------------------------
# 3️⃣ Multi-label model
# ----------------------------
def train_multilabel_model():
    print("\n[Multi-label] Training Diagnostic Model...")

    # Load dataset
    df = pd.read_csv(DIAGNOSTIC_CSV)
    df["diagnoses"] = df["diagnoses"].apply(eval)

    # Features and labels
    X = df.drop(columns=["diagnoses"])
    y_raw = df["diagnoses"]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n📋 Classification Report (Multi-label):")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

    # Save model + encoder
    dump(model, os.path.join(MODEL_DIR, "diagnostic_model.pkl"))
    dump(mlb, os.path.join(MODEL_DIR, "label_binarizer.pkl"))
    print("✅ Multi-label model saved to artifacts/diagnostic_model.pkl")

# ----------------------------
# 4️⃣ Run everything
# ----------------------------
if __name__ == '__main__':
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"{LABELS_FILE} not found. Generate it first from predictions.csv.")

    # Load and train traditional models
    df_labels = pd.read_csv(LABELS_FILE)
    df_features = df_labels.drop(columns=['label', 'abnormalities'], errors='ignore')
    train_unsupervised(df_features)
    train_supervised(df_labels)

    # Train new AI-powered diagnostic model
    train_multilabel_model()

    print("\n✅ All models trained & saved successfully!")
