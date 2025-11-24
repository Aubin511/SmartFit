import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import numpy as np

# ================= CONFIGURATION =================
DATA_DIR = "data_split/val"
MODEL_PATH = "backend/finetuned_model.pth"
CLASSES_PATH = "backend/classes.txt"
BATCH_SIZE = 32
# =================================================

def evaluate():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Test sur : {device}")

    # 2. Recharger les transformations (Doit être identique à l'entraînement, sans augmentation)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Charger les données de test
    if not os.path.exists(DATA_DIR):
        print(f"Dossier {DATA_DIR} introuvable.")
        return

    test_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Récupérer les noms des classes depuis le dossier (ou le fichier txt)
    class_names = test_dataset.classes
    print(f"Classes : {class_names}")

    # 4. Recharger le modèle (Architecture + Poids)
    print("Chargement du modèle...")
    model = models.resnet18(weights=None) # On ne charge pas ImageNet, on va charger NOS poids
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Charger tes poids sauvegardés
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Modèle non trouvé !")
        return

    model = model.to(device)
    model.eval() # IMPORTANT

    # 5. Boucle de prédiction
    all_preds = []
    all_labels = []

    print("Calcul des prédictions...")
    with torch.no_grad(): # Pas besoin de gradients pour le test (économie mémoire)
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Métriques
    # Calculer l'accuracy
    correct_predictions = np.sum(np.array(all_preds) == np.array(all_labels))
    accuracy = correct_predictions / len(all_labels)
    print(f"\n📊 Précision Globale (Accuracy) : {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate()