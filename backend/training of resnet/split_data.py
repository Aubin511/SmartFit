import os
import shutil
import random

SOURCE_DIR = "data/base"
OUTPUT_DIR = "data_split"

TRAIN_RATIO = 0.80 
VAL_RATIO   = 0.10 
TEST_RATIO  = 0.10

def split_dataset():
    print(f"Data spliting")
    if not os.path.exists(SOURCE_DIR):
        print(f"Erreur : Le dossier source '{SOURCE_DIR}' n'existe pas.")
        return

    # Créer les dossiers de destination
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(OUTPUT_DIR, split)
        if os.path.exists(split_path):
            print(f"Attention : Le dossier '{split_path}' existe déjà. Suppression du dossier existant")
            shutil.rmtree(split_path)
        os.makedirs(split_path)

    # Liste des catégories (T-shirt, Jean, etc.)
    categories = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    total_images = 0

    for category in categories:
        # Chemin source de la catégorie
        src_cat_path = os.path.join(SOURCE_DIR, category)
        
        # Récupérer toutes les images
        images = [f for f in os.listdir(src_cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images) # Mélange aléatoire important !
        
        # Calcul des index de coupure
        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        # Découpage des listes
        train_imgs = images[:train_end]
        val_imgs   = images[train_end:val_end]
        test_imgs  = images[val_end:]
        
        # Dictionnaire pour boucler facilement
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }

        print(f"Traitement de '{category}': {n} images -> Train:{len(train_imgs)}, Val:{len(val_imgs)}, Test:{len(test_imgs)}")

        # Copie des fichiers
        for split_name, img_list in splits.items():
            dest_cat_path = os.path.join(OUTPUT_DIR, split_name, category)
            os.makedirs(dest_cat_path, exist_ok=True)
            
            for img in img_list:
                shutil.copy2(
                    os.path.join(src_cat_path, img),
                    os.path.join(dest_cat_path, img)
                )
                total_images += 1

    print("-" * 30)
    print(f"Terminé ! {total_images} images réparties dans '{OUTPUT_DIR}/'.")
    print(f"Structure créée :")
    print(f"   - {OUTPUT_DIR}/train (pour train_model.py)")
    print(f"   - {OUTPUT_DIR}/val   (pour vérifier pendant l'entraînement)")
    print(f"   - {OUTPUT_DIR}/test  (pour evaluate.py)")

if __name__ == "__main__":
    split_dataset()