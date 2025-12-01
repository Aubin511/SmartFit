from flask import Flask, render_template, request, redirect, url_for, flash
import os

# Tes imports existants
from backend.model import detect_clothing, save_image
from backend.database import db, Clothing, init_db_app 

app = Flask(__name__)

# --- CONFIGURATION (AJOUTÉE) ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///macollection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# AJOUT CRUCIAL : Clé secrète pour sécuriser les sessions et permettre l'utilisation des messages flash.
app.config['SECRET_KEY'] = 'ta_super_cle_secrete_ici_meme'

# init database
init_db_app(app) 

# dossier de sauvegarde des images uploadées
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#homepage
@app.route('/') 
def homepage():
    return render_template('homepage.html')

# Upload page (Étape 1 : Téléchargement et Détection)
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Lors d'un POST (téléchargement d'une image)
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            # 1. Sauvegarder l'image
            filepath = save_image(file)
            
            # 2. Détecter la catégorie
            category = detect_clothing(filepath)
            
            # 3. ON N'ENREGISTRE PAS EN BASE ICI. 
            # On renvoie les données pour la confirmation.
            return render_template(
                'upload.html',
                predicted_category=category,
                filepath=filepath,
                show_confirmation=True # Indique au template d'afficher le formulaire de confirmation
            )
        else:
            flash("Veuillez sélectionner un fichier à télécharger.", "error")
            return redirect(url_for('upload'))
            
    # Pour un GET ou après une erreur, on affiche le formulaire de base
    return render_template('upload.html', show_confirmation=False)

# Nouvelle Route de Confirmation (Étape 2 : Validation et Enregistrement)
@app.route('/confirm_add', methods=['POST'])
def confirm_add():
    # 1. Récupérer les données du formulaire de confirmation
    final_category = request.form.get('final_category') # La catégorie potentiellement modifiée
    filepath = request.form.get('filepath')             # Le chemin de l'image

    if not final_category or not filepath:
        flash("Erreur: Données de confirmation incomplètes.", "error")
        return redirect(url_for('upload'))
    
    # 2. SAUVEGARDE EN SQL
    try:
        new_item = Clothing(image_path=filepath, category=final_category)
        db.session.add(new_item)
        db.session.commit()
        
        flash(f"Le vêtement '{final_category}' a été ajouté à la garde-robe!", "success")
        return redirect(url_for('wardrobe'))

    except Exception as e:
        flash(f"Erreur lors de l'enregistrement en base de données: {e}", "error")
        return redirect(url_for('upload'))


@app.route('/wardrobe')
def wardrobe():
    all_clothes = Clothing.query.all()
    return render_template('wardrobe.html', clothes=all_clothes)

if __name__ == '__main__':
    app.run(debug=True)
