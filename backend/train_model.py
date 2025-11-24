import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
# train set of data
DATA_DIR = "data_split/train" 

# Dossier où on va sauvegarder le modèle entraîné
SAVE_DIR = "backend"
MODEL_FILENAME = "finetuned_model.pth"
CLASSES_FILENAME = "classes.txt"

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10     
LEARNING_RATE = 0.001

def train():
    print(f"--- Start training ---")

    # ---------------------------------------------------------
    # 2. DÉTECTION DU MATÉRIEL (GPU / CPU)
    # ---------------------------------------------------------
    # Utilise la carte graphique si possible (Nvidia CUDA ou Mac MPS), sinon le processeur
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Matériel utilisé : {device}")

    # Data transormation for Resnet
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),      #Resnet input
        #transforms.RandomHorizontalFlip(),  # Data augmentation TO ACTIVATE LATER
        transforms.ToTensor(),              
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet Standardisation
    ])

    # Vérification que le dossier existe
    if not os.path.exists(DATA_DIR):
        print(f"Erreur : Le dossier {DATA_DIR} n'existe pas.")
        return

    # Upload homemade dataset
    image_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
    
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    class_names = image_dataset.classes
    num_classes = len(class_names)
    dataset_size = len(image_dataset)

    print(f"Dataset size : {dataset_size} images")
    print(f"Classes detected ({num_classes}) : {class_names}")

    # Transfer learning from Resnet18
    print("Upload Pretrained Resnet18...")    
    # upload model with wieghts from ImageNet
    model = models.resnet18(weights='IMAGENET1K_V1')

    # AVOID COMPLETELY NEW LEARNING
    for param in model.parameters():
        param.requires_grad = False

    # REPLACE HEAD WITH OUR CATEGORIES
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    # ---------------------------------------------------------
    # TRAINING PART
    # ---------------------------------------------------------
    # Fonction de perte (CrossEntropy est standard pour la classification)
    criterion = nn.CrossEntropyLoss()

    # Optimiseur : Seule la dernière couche (model.fc) sera mise à jour !
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------------------
    # 6. BOUCLE D'ENTRAÎNEMENT
    # ---------------------------------------------------------
    since = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        model.train()  # Met le modèle en mode entraînement
        
        running_loss = 0.0
        running_corrects = 0

        # Boucle sur les lots d'images (batches)
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Remise à zéro des gradients
            optimizer.zero_grad()

            # Forward (passage avant)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward (rétropropagation) + Optimisation
            loss.backward()
            optimizer.step()

            # Statistiques
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.float() / dataset_size

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'\n✅ Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Précision finale : {epoch_acc:.4f}')

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    save_path = os.path.join(SAVE_DIR, MODEL_FILENAME)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved in : {save_path}")

    # Sauvegarde des noms de classes (important pour retrouver 'T-shirt' depuis l'ID 0)
    classes_path = os.path.join(SAVE_DIR, CLASSES_FILENAME)
    with open(classes_path, "w") as f:
        for c in class_names:
            f.write(f"{c}\n")
    print(f"Classes sauvegardées sous : {classes_path}")

if __name__ == "__main__":
    train()