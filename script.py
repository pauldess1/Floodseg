import os
import shutil

# Répertoire contenant les fichiers de sortie existants
output_dir = "/home/pdessain/Bureau/Semantic_seg/Exemples/masks+slic"
# Répertoire contenant les images d'entrée
image_dir = "/home/pdessain/Bureau/Semantic_seg/FloodNet_images/queries"
# Répertoire de destination pour les fichiers copiés
dest_dir = "/home/pdessain/Bureau/Semantic_seg/Exemples/queries"

# Créer le répertoire de destination s'il n'existe pas
os.makedirs(dest_dir, exist_ok=True)

# Parcourir chaque fichier de sortie existant
for output_file in os.listdir(output_dir):
    if output_file.endswith("_mask_slic.png"):
        # Extraire l'identifiant xxxxx
        basename = output_file[:-14]  # Enlever "_mask_slic.png"
        identifier = basename.split('_')[1]
        
        # Construire le chemin du fichier d'image d'entrée correspondant
        img_path = os.path.join(image_dir, f"query_{identifier}.png")
        
        # Vérifier si l'image d'entrée existe
        if os.path.isfile(img_path):
            # Copier l'image d'entrée vers le répertoire de destination
            shutil.copy(img_path, dest_dir)
        else:
            print(f"Image d'entrée {img_path} introuvable, sautée.")
