import os

# dictionary of files and their content
files = {
    "README.md": """# AgenticSprint - AI Diagnostic Assistant (Demo)

Run locally:
1. python -m venv .venv
2. source .venv/bin/activate  (or .venv\\Scripts\\activate on Windows)
3. pip install -r requirements.txt
4. python app.py
5. Open http://127.0.0.1:5000/
""",

    "requirements.txt": """flask
pdfplumber
pandas
numpy
scikit-learn
joblib
shap
plotly
python-dotenv
""",

    "app.py": """from flask import Flask, render_template, request, redirect, url_for, jsonify
from ingestion.parse_pdf import parse_pdf_to_dict
from advisory.explain import explain_for_patient
from analysis.model import predict_supervised
from analysis.rules import simple_abnormalities
from alert.redflags import detect_red_flags
import os
from datetime import datetime
import csv
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'data/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)

def log_prediction(patient_id, result):
    fname = 'logs/predictions.csv'
    header = ['timestamp','patient_id','result_json']
    if not os.path.exists(fname):
        with open(fname,'w',newline='') as f:
            writer = csv.writer(f); writer.writerow(header)
    with open(fname,'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), patient_id, json.dumps(result)])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f:
        return 'No file', 400
    save_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(save_path)
    return redirect(url_for('dashboard', filename=f.filename))

@app.route('/dashboard')
def dashboard():
    filename = request.args.get('filename')
    if not filename: return redirect(url_for('index'))
    path = os.path.join(UPLOAD_FOLDER, filename)
    data = parse_pdf_to_dict(path)
    rules = simple_abnormalities(data)
    alerts = detect_red_flags(data)
    model_out = predict_supervised(data)
    explanation = explain_for_patient(data, model_ranked=model_out)
    log_prediction(filename, explanation)
    return render_template('dashboard.html', patient=data, rules=rules, alerts=alerts, explanation=explanation)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    patient = request.json
    model_out = predict_supervised(patient)
    explanation = explain_for_patient(patient, model_ranked=model_out)
    return jsonify(explanation)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
""",

    "train.py": """import pandas as pd
from analysis.model import train_unsupervised
import os

if __name__ == '__main__':
    if not os.path.exists('data/labels.csv'):
        df = pd.DataFrame({
            'WBC': [8000, 14000, 17000, 7000, 13000],
            'CRP': [5, 60, 150, 3, 40],
            'Hemoglobin': [13, 9, 8, 14, 11],
            'Platelet': [200000, 90000, 110000, 250000, 150000],
            'label': ['normal','infection','sepsis','normal','infection']
        })
        df.to_csv('data/labels.csv', index=False)
    df = pd.read_csv('data/labels.csv')
    y = df['label']; X = df.drop(columns=['label'])
    train_unsupervised(X)
    print("Unsupervised IsolationForest model trained.")
""",

    "run_demo.py": """print('This can be used later for scripted demo playback.')""",

    "ingestion/__init__.py": "",
    "ingestion/parse_pdf.py": """import pdfplumber, re
from pathlib import Path
import pandas as pd

LAB_KEYS = ['Hemoglobin','Hb','RBC','WBC','WBC Count','Neutrophils','Lymphocytes',
    'Platelet','Platelet Count','MPV','MCV','MCH','Hematocrit','RDW',
    'Fasting Blood Sugar','Glucose','HbA1c','Cholesterol','Triglyceride',
    'HDL','LDL','Creatinine','Urea','SGPT','SGOT','Sodium','Potassium',
    'Vitamin D','Vitamin B12','IgE','CRP']

def extract_numbers_from_line(line): return re.findall(r'(?<!\\d)[+-]?\\d+(?:\\.\\d+)?', line)

def parse_pdf_to_dict(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages: text.append(page.extract_text() or "")
    txt = "\\n".join(text); lines = [l.strip() for l in txt.splitlines() if l.strip()]
    data = {}
    for key in LAB_KEYS:
        for line in lines:
            if key.lower() in line.lower():
                nums = extract_numbers_from_line(line)
                if nums:
                    try: data[key] = float(nums[0])
                    except: pass
                break
    m = re.search(r'(Male|Female|M|F)[,\\s/]+(\\d{1,3})', txt, re.IGNORECASE)
    if m: data['sex'], data['age'] = m.group(1), int(m.group(2))
    data['source'] = Path(pdf_path).name
    return data

def pdfs_to_dataframe(folder='data/reports'):
    folder = Path(folder)
    return pd.DataFrame([parse_pdf_to_dict(p) for p in folder.glob('*.pdf')])
""",

    "analysis/__init__.py": "",
    "analysis/model.py": """import os, numpy as np
from joblib import dump, load
from sklearn.ensemble import IsolationForest
MODEL_DIR = 'analysis/artifacts'; os.makedirs(MODEL_DIR, exist_ok=True)

def train_unsupervised(X):
    X = X.fillna(X.median())
    iso = IsolationForest(contamination=0.05, random_state=42); iso.fit(X)
    dump({'iso': iso, 'columns': list(X.columns)}, os.path.join(MODEL_DIR,'iso.joblib'))
    return iso

def load_unsupervised():
    try:
        obj = load(os.path.join(MODEL_DIR,'iso.joblib'))
        return obj['iso'], obj['columns']
    except: return None, None

def predict_anomaly_score(sample_dict):
    iso, cols = load_unsupervised(); 
    if iso is None: return None
    import pandas as pd
    x = pd.DataFrame([sample_dict]).reindex(columns=cols).fillna(0)
    score = iso.decision_function(x)[0]; return float(score)

def predict_supervised(sample_dict):
    try:
        obj = load(os.path.join(MODEL_DIR,'rf.joblib'))
        model, imputer, cols = obj['model'], obj['imputer'], list(obj['model'].feature_names_in_)
        import pandas as pd
        x = pd.DataFrame([sample_dict]).reindex(columns=cols)
        Ximp = imputer.transform(x)
        probs = model.predict_proba(Ximp)[0]; classes = model.classes_
        return [{'diagnosis':c,'prob':float(p)} for c,p in zip(classes,probs)]
    except:
        s = predict_anomaly_score(sample_dict)
        return [{'diagnosis':'Anomalous profile','score':s}] if s else None
""",

    "analysis/rules.py": """def check_sepsis_like(data):
    reasons=[]; score=0
    wbc = data.get('WBC') or data.get('WBC Count') or 0
    crp = data.get('CRP') or 0
    if wbc>12000: reasons.append('High WBC'); score+=1
    if crp>50: reasons.append('High CRP'); score+=0.8
    risk='LOW'
    if score>=1: risk='HIGH'
    elif score>=0.5: risk='MEDIUM'
    return {'risk':risk,'reasons':reasons,'score':score}

def simple_abnormalities(data):
    abn=[]
    if (data.get('Hemoglobin') or data.get('Hb') or 99)<10: abn.append('Anemia (low Hb)')
    if (data.get('Platelet') or data.get('Platelet Count') or 999999)<100000: abn.append('Thrombocytopenia')
    if (data.get('Vitamin D') or 999)>0 and data.get('Vitamin D',999)<20: abn.append('Vitamin D deficiency')
    return abn
""",

    "advisory/__init__.py": "",
    "advisory/explain.py": """from analysis import rules
from alert import redflags
def explain_for_patient(data, model_ranked=None):
    return {
        'rule_risk': rules.check_sepsis_like(data),
        'abnormalities': rules.simple_abnormalities(data),
        'alerts': redflags.detect_red_flags(data),
        'model_ranked': model_ranked,
        'evidence_snapshot': {k:v for k,v in data.items() if isinstance(v,(int,float))}
    }
""",

    "alert/__init__.py": "",
    "alert/redflags.py": """def detect_red_flags(data):
    alerts=[]
    wbc = data.get('WBC') or data.get('WBC Count') or 0
    crp = data.get('CRP') or 0
    glucose = data.get('Fasting Blood Sugar') or data.get('Glucose') or 0
    if wbc>15000 or crp>100:
        alerts.append({'title':'Possible Severe Infection / Sepsis','urgency':'CRITICAL',
            'why':f'WBC={wbc}, CRP={crp}','recommendation':'Immediate escalation to ER'})
    if glucose>400:
        alerts.append({'title':'Hyperglycemic emergency','urgency':'CRITICAL',
            'why':f'Glucose={glucose}','recommendation':'Urgent insulin management'})
    return alerts
""",

    "templates/index.html": """<!doctype html>
<html><head>
<title>AgenticSprint - Upload Report</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="/static/css/styles.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-5">
<div class="card shadow-sm mx-auto" style="max-width:900px; border-radius:12px;">
<div class="card-body">
<h3 class="card-title">AgenticSprint — AI Diagnostic Assistant (Demo)</h3>
<p class="text-muted">Upload a lab report PDF to run ingestion, rules-based checks, and explainable alerts.</p>
<form action="/upload" method="post" enctype="multipart/form-data">
<input class="form-control mb-3" type="file" name="file" accept=".pdf" required>
<button class="btn btn-primary">Upload & Analyze</button>
</form>
</div></div></div></body></html>
""",

    "templates/dashboard.html": """<!doctype html>
<html><head>
<title>Dashboard - AgenticSprint</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="/static/css/styles.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-light"><div class="container my-4">
<div class="d-flex justify-content-between align-items-center mb-3">
<h2>Patient Analysis</h2><a href="/" class="btn btn-outline-secondary">Upload another</a>
</div>
<div class="row g-3">
<div class="col-md-5"><div class="card p-3">
<h5>Compact Summary</h5>
<table class="table table-sm"><tbody>
{% for k,v in patient.items() %}<tr><th>{{k}}</th><td>{{v}}</td></tr>{% endfor %}
</tbody></table></div>
<div class="card p-3 mt-3"><h5>Abnormalities & Rules</h5>
{% if rules %}<ul>{% for r in rules %}<li>{{r}}</li>{% endfor %}</ul>{% else %}
<div class="text-success">No abnormalities</div>{% endif %}</div></div>
<div class="col-md-7"><div class="card p-3 mb-3"><h5>Alerts</h5>
{% if alerts %}{% for a in alerts %}
<div class="alert alert-danger"><b>{{a.title}}</b> <span class="badge bg-danger">{{a.urgency}}</span>
<p><small>{{a.why}}</small></p><em>{{a.recommendation}}</em></div>
{% endfor %}{% else %}<div class="text-success">No urgent alerts detected.</div>{% endif %}
</div>
<div class="card p-3"><h5>Explainability & Evidence</h5>
<pre style="max-height:250px;overflow:auto;">{{ explanation | tojson(indent=2) }}</pre></div></div>
</div></div></body></html>
""",

    "static/css/styles.css": "body{font-family:system-ui;} .card{border-radius:12px;}",
    "static/js/dashboard.js": "console.log('Dashboard script loaded');",
    "data/reports/.gitkeep": "",
}

# create files
for path, content in files.items():
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Full project created successfully!")
