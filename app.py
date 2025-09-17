from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from ingestion.parse_pdf import parse_pdf_to_dict
from advisory.explain import explain_for_patient, generate_summary
from analysis.model import predict_supervised
from analysis.rules import simple_abnormalities
from alert.redflags import detect_red_flags

import os, csv, json
from datetime import datetime

app = Flask(__name__)

# ---------------- Config ----------------
app.config['SECRET_KEY'] = 'dev-secret-change-me'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = 'data/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ---------------- DB & Login ----------------
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ---------------- User Model ----------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(180), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- Logging Predictions ----------------
def log_prediction(patient_id, result):
    fname = 'logs/predictions.csv'
    header = ['timestamp', 'patient_id', 'result_json']
    if not os.path.exists(fname):
        with open(fname, 'w', newline='') as f:
            csv.writer(f).writerow(header)
    with open(fname, 'a', newline='') as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(), patient_id, json.dumps(result)])

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        password = request.form.get('password', '')
        if not name or not password:
            flash('Name and password are required', 'warning')
            return redirect(url_for('signup'))

        if User.query.filter_by(name=name).first():
            flash('Name already registered', 'danger')
            return redirect(url_for('signup'))

        u = User(name=name)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        password = request.form.get('password', '')
        u = User.query.filter_by(name=name).first()
        if u and u.check_password(password):
            login_user(u)
            flash('Logged in', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid credentials', 'danger')
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out', 'info')
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    f = request.files.get('file')
    symptoms = request.form.get('symptoms', '')
    medications = request.form.get('medications', '')

    if not f:
        flash('No file selected', 'warning')
        return redirect(url_for('dashboard'))

    save_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(save_path)

    # Save everything in session
    session['uploaded_file'] = f.filename
    session['symptoms'] = symptoms
    session['medications'] = medications

    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    filename = session.pop('uploaded_file', None) or request.args.get('filename')
    if not filename:
        flash('No report uploaded yet', 'info')
        return render_template(
            'dashboard.html',
            patient=None, rules=None, alerts=None, explanation=None,
            symptoms=None, medications=None, summary=None
        )

    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        flash('File not found', 'danger')
        return render_template(
            'dashboard.html',
            patient=None, rules=None, alerts=None, explanation=None,
            symptoms=None, medications=None, summary=None
        )

    # ✅ Get symptoms and meds from session
    symptoms = [s.strip() for s in session.pop('symptoms', '').split(',') if s.strip()]
    medications = [m.strip() for m in session.pop('medications', '').split(',') if m.strip()]

    # ✅ Parse PDF with extra inputs
    data = parse_pdf_to_dict(path, symptoms=symptoms, medications=medications)

    rules = simple_abnormalities(data)
    alerts = detect_red_flags(data)

    try:
        model_out = predict_supervised(data)
    except Exception as e:
        print(f"[ERROR] Supervised model failed: {e}")
        model_out = None

    explanation = explain_for_patient(data, model_ranked=model_out)
    summary = generate_summary(data, explanation)   # ✅ Clinical summary section
    log_prediction(filename, explanation)

    return render_template(
        'dashboard.html',
        patient=data,
        rules=rules,
        alerts=alerts,
        explanation=explanation,
        summary=summary,
        symptoms=symptoms,
        medications=medications
    )

@app.route('/api/analyze', methods=['POST'])
@login_required
def api_analyze():
    patient = request.json
    model_out = predict_supervised(patient)
    explanation = explain_for_patient(patient, model_ranked=model_out)
    return jsonify(explanation)

# ---------------- Run App ----------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=2000)
