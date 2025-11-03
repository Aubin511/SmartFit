from flask import Flask, render_template, request, redirect, url_for
import os
from backend.model import detect_clothing

app = Flask(__name__)

# dossier de sauvegarde des images uploadées
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/') #homepage
def home():
    return render_template('homepage.html')

if __name__ == '__main__':
    app.run(debug=True)
