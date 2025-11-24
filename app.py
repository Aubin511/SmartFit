from flask import Flask, render_template, request, redirect, url_for
import os
from backend.model_resnet18 import Clothing, detect_clothing, add_to_wardrobe_json, save_image

app = Flask(__name__)

# dossier de sauvegarde des images uploadées
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#homepage
@app.route('/') 
def homepage():
    return render_template('homepage.html')

#Upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    category = None
    filepath = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file :
            filepath = save_image(file)
            category = detect_clothing(filepath)
            cloth = Clothing(filepath, category)
            add_to_wardrobe_json(cloth)
    return render_template('upload.html',predicted_category=category, filepath=filepath)

@app.route('/wardrobe')
def wardrobe():
    return render_template('wardrobe.html')

if __name__ == '__main__':
    app.run(debug=True)
