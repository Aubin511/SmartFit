import torch
from torchvision import models, transforms
from PIL import Image
import os

# CONFIG
IMG_PATH = "static/uploads/51bdf9e1-6ff4-46fe-9051-2821fff88f85.jpg" # Mets une image ici pour tester
MODEL_PATH = "backend/finetuned_model_80_24112025.pth"
CLASSES_PATH = "backend/classes.txt"

def predict_single_image(img_path,model_path,classes_path):
    # Upload classes
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Model config
    device = torch.device("cpu") # enough for one image
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Préparer l'image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add dimension batch (1, 3, 224, 224)

    # Predict part
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get first prediction
        conf, predicted_idx = torch.max(probabilities, 0)
        print(f"Image : {IMG_PATH}")

        # Vérifier la condition (0.5 correspond à 50%)
        if conf.item() < 0.5:
            print("Confiance faible, récupération du Top 3...")
            
            # Récupère les 3 meilleures probabilités et leurs indices
            top_probs, top_idxs = torch.topk(probabilities, 3)
            
            # On crée une liste de tuples (Nom de la classe, Probabilité)
            top_3_results = []
            for i in range(3):
                idx = top_idxs[i].item()
                prob = top_probs[i].item()
                label = class_names[idx]
                top_3_results.append((label, prob))
                
            # Exemple d'utilisation des résultats
            print(top_3_results) 
            # Sortie : [('Chat', 0.45), ('Chien', 0.30), ('Oiseau', 0.15)]

        else:
            # Si la confiance est suffisante, on garde juste le Top 1
            predicted_label = class_names[predicted_idx]
            print(f"Prédiction : {predicted_label}")
            print(f"Confiance : {conf.item() * 100:.2f}%")

if __name__ == "__main__":
    predict_single_image(IMG_PATH,MODEL_PATH,CLASSES_PATH)