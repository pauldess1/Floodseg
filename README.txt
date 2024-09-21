Projet : Segmentation d'Images avec PSPNet et SLIC

Ce projet utilise un modèle de segmentation d'image (PSPNet) et l'algorithme SLIC (Simple Linear Iterative Clustering) pour segmenter des images et ajuster les segments de manière cohérente. Le script prend une image en entrée, la segmente, applique SLIC pour obtenir des superpixels, et ajuste les classes de ces superpixels en fonction de la classe prédominante dans chaque cluster.
Toutes les images à segmenter sont d'abord transformées en images carrées en redimmensionnant en un carré de côté max(height, width)

Structure du Projet

    segmentation/inference.py : Contient les fonctions pour charger le modèle et faire des inférences sur les images.
    slic/slic.py : Contient l'implémentation de l'algorithme SLIC pour générer les superpixels.

Usage

Pour exécuter le script, utilisez la commande suivante :

python run.py --model_path <chemin_du_modèle> --img_path <chemin_image> --out_path_array <chemin_sortie_array> --out_path_image1 <chemin_sortie_image1> --slic_contours_path <chemin_contours_slic> --slic_clusters_path <chemin_clusters_slic> --out_path_image2 <chemin_sortie_image2>

Arguments

    --model_path : Chemin vers le fichier du modèle pré-entraîné PSPNet.
    --img_path : Chemin vers l'image d'entrée à segmenter.
    --out_path_array : Chemin pour sauvegarder le masque de segmentation en format numpy.
    --out_path_image1 : Chemin pour sauvegarder le masque de segmentation en format image.
    --slic_contours_path : Chemin pour sauvegarder les contours générés par SLIC. (format numpy)
    --slic_clusters_path : Chemin pour sauvegarder les clusters générés par SLIC. (format numpy)
    --out_path_image2 : Chemin pour sauvegarder le masque final avec ajustement par slic


Le script subpro.py permet de générer les segmentations pour l'ensemble de requêtes. 

On peut aussi tenter de réentrainer le modèle en utilisant le script segmentation/training.py et les données d'entrainement issues de FloodNet. 
