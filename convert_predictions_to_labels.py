import os
import pandas as pd
import json

# Input/Output files
PRED_FILE = "logs/predictions.csv"
LABELS_FILE = "data/labels4.csv"

# Load predictions
df = pd.read_csv(PRED_FILE)

rows = []
for _, row in df.iterrows():
    data = json.loads(row["result_json"])

    # Extract features
    snapshot = data.get("evidence_snapshot", {})
    
    # Risk level = classification label
    risk = data.get("rule_risk", {}).get("risk", "UNKNOWN")

    # Abnormalities list → string
    abnormalities = ",".join(data.get("abnormalities", []))

    # Build row
    snapshot["label"] = risk   # supervised learning target
    snapshot["abnormalities"] = abnormalities  # extra info for explainability
    rows.append(snapshot)

# Create DataFrame
df_out = pd.DataFrame(rows)

# Save
os.makedirs("data", exist_ok=True)
df_out.to_csv(LABELS_FILE, index=False)

print(f"✅ labels4.csv saved at {LABELS_FILE}")
print("Columns:", df_out.columns.tolist())
