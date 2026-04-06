from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import io, base64, os, uuid

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = "secret123"

# ================= DATABASE =================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
sns.set_style("whitegrid")

# ================= USER TABLE =================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Create DB
with app.app_context():
    db.create_all()

# ================= FOLDERS =================
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# ================= LOGIN =================
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            error = "❌ All fields required!"
            return render_template('login.html', error=error)

        user = User.query.filter_by(username=username).first()

        # LOGIN
        if user:
            if user.password != password:
                error = "❌ Incorrect password!"
            else:
                session['user'] = user.username
                session['email'] = user.email
                return redirect(url_for('home'))

        # SIGNUP
        else:
            new_user = User(username=username, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            session['user'] = username
            session['email'] = email
            return redirect(url_for('home'))

    return render_template('login.html', error=error)

# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ================= HOME =================
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['user'])

# ================= DOWNLOAD =================
@app.route('/download/<filename>')
def download(filename):
    if 'user' not in session:
        return redirect(url_for('login'))
    return send_from_directory(DOWNLOAD_FOLDER, filename, as_attachment=True)

# ================= AI PREDICTION =================
@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        file = request.files['file']
        if file.filename == '':
            return "❌ No file selected"

        df = pd.read_csv(file)

        # ---------- CLEANING ----------
        if 'customerID' in df.columns:
            df.drop(['customerID'], axis=1, inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        if 'Churn' not in df.columns:
            return "❌ 'Churn' column not found"

        if df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df.dropna(subset=['Churn'], inplace=True)
        df['Churn'] = df['Churn'].astype(int)

        # Encode categorical
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes

        # ---------- MODEL ----------
        X = df.drop('Churn', axis=1)
        y = df['Churn']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=300)
        model.fit(X_scaled, y)

        predictions = model.predict(X_scaled)
        score = model.score(X_scaled, y)

        df['Prediction'] = ["Churn" if p == 1 else "No Churn" for p in predictions]

        churn = (predictions == 1).sum()
        total = len(predictions)

        # ---------- SAVE RESULT FOR DOWNLOAD ----------
        unique_filename = f"{session['user']}_prediction_{uuid.uuid4().hex}.csv"
        output_file = os.path.join(DOWNLOAD_FOLDER, unique_filename)
        df.to_csv(output_file, index=False)

        download_link = url_for('download', filename=unique_filename)

        # ---------- GRAPH ----------
        plt.figure(figsize=(5, 4))
        sns.countplot(x=predictions)
        plt.title("Churn Prediction")
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        graph1 = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # ---------- HEATMAP ----------
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, cmap='coolwarm', annot=False)
        img2 = io.BytesIO()
        plt.savefig(img2, format='png', bbox_inches='tight')
        img2.seek(0)
        graph2 = base64.b64encode(img2.getvalue()).decode()
        plt.close()

        # ---------- TABLE ----------
        table = df.head(100).to_html(index=False)

        return render_template(
            'result.html',
            username=session['user'],
            total=total,
            churn=churn,
            score=round(score * 100, 2),
            graph1=graph1,
            graph2=graph2,
            tables=[table],
            summary_only=False,
            download_link=download_link
        )

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ================= SUMMARY =================
@app.route('/summary', methods=['POST'])
def summary():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        file = request.files['file']
        df = pd.read_csv(file)

        if 'Churn' not in df.columns:
            return "❌ 'Churn' column missing"

        if df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df.dropna(subset=['Churn'], inplace=True)

        total = len(df)
        churn = (df['Churn'] == 1).sum()
        connected = total - churn

        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        return render_template(
            'result.html',
            username=session['user'],
            total=total,
            churn=churn,
            connected=connected,
            columns=columns,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            summary_only=True
        )

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)