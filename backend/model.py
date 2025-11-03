CATEGORIES = [
    "T-shirt",
    "Chemise",
    "Pull",
    "Pantalon",
    "Jean",
    "Jupe",
    "Robe",
    "Veste",
    "Manteau",
    "Chaussures"
]

class Clothing:
    def __init__(self, image_path, category = None):
        self.category = category
        self.image_path = image_path
    def __repr__(self): #used with print command
        return f"{self.category}"
import random

def detect_clothing(image_path):
    """
    function that takes an image as an input and returns the predicted type of cloth
    """
    # Choisir un type de vêtement aléatoire
    category = random.choice(CATEGORIES)
    return category

# ==========================
# wardrobe storage
# ==========================
wardrobe = []
wardrobe_dict = {}

def add_to_wardrobe(cloth):
    wardrobe.append(cloth)
    wardrobe_dict[cloth.image_path] = cloth.category
    return ()

image_path = "static/uploads/cloth_1.jpg"
predicted_clothing = detect_clothing(image_path)
print(predicted_clothing)
clothing = Clothing(image_path)
clothing.category = predicted_clothing
print(clothing)
add_to_wardrobe(clothing)
print(wardrobe_dict)

import os
import json

UPLOAD_FOLDER = "static/uploads/"
results_file = "static/wardrobe_results.json"
wardrobe_data = []

for cloth in wardrobe :
        wardrobe_data.append({
            "image": cloth.image_path,
            "category": cloth.category
        })

# Sauvegarde en JSON
with open(results_file, "w") as f:
    json.dump(wardrobe_data, f, indent=4)

print("Données wardrobe sauvegardées dans wardrobe_results.json")