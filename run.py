import numpy as np
from collections import Counter
from PIL import Image
from segmentation.inference import load_model, infer_image, label_to_rgb
from slic.slic import slic
import argparse
import torch
import os

def principale_classe(cluster, segmentation):
    classes = []
    for pixel in cluster :
        classes.append(segmentation[pixel[0]][pixel[1]])
    return Counter(classes).most_common(1)[0][0]

def change_classe(cluster, segmentation, new_image):
    princip = principale_classe(cluster, segmentation)
    for pixel in cluster :
        new_image[pixel[0], pixel[1]]=princip
    return new_image

def save_mask(mask, contours, out_path_image2):
    color_mask = label_to_rgb(mask)
    for i in range(len(contours)):
            color_mask[contours[i][0], contours[i][1]] = (0, 0, 0)
    color_mask = Image.fromarray(color_mask)
    color_mask.save(os.path.join(out_path_image2))

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = args.model_path
    img_path = args.img_path
    out_path_array = args.out_path_array
    out_path_image1 = args.out_path_image1
    slic_contours_path = args.slic_contours_path
    slic_clusters_path = args.slic_clusters_path
    out_path_image2 = args.out_path_image2

    ### Load model ###
    model = load_model(device, model_path)
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    size = (max(height, width), max(height, width))
    
    ### Segmentation ###
    
    infer_image(img_path, out_path_array, out_path_image1, model, device, size)
    segmentation = np.load(out_path_array)
    new_image = segmentation.copy()

    ### SLIC ###
    slic(img_path, slic_contours_path, slic_clusters_path, size)
    clusters = np.load(slic_clusters_path, allow_pickle=True)
    contours = np.load(slic_contours_path, allow_pickle=True)

    for cluster in clusters:
        new_image = change_classe(cluster, segmentation, new_image)

    save_mask(new_image, contours, out_path_image2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation d'image avec modèle PSPNet et SLIC.")
    
    parser.add_argument('--model_path', type=str, default="segmentation/weights/pspnet_best.pth",
                        help='Chemin vers le modèle pré-entraîné.')
    parser.add_argument('--img_path', type=str, default='/home/pdessain/Bureau/VPR/Datasets/Dataset/Query_Images/IDS_Image_00044.png',
                        help='Chemin vers l\'image d\'entrée.')
    parser.add_argument('--out_path_array', type=str, default="Results/arrays/mask.npy",
                        help='Chemin vers le fichier de sortie pour le masque en format numpy.')
    parser.add_argument('--out_path_image1', type=str, default="Results/masks/mask.png",
                        help='Chemin vers le fichier de sortie pour le masque en format image.')
    parser.add_argument('--slic_contours_path', type=str, default="Results/arrays/contours.npy",
                        help='Chemin vers le fichier de contours SLIC.')
    parser.add_argument('--slic_clusters_path', type=str, default="Results/arrays/clusters.npy",
                        help='Chemin vers le fichier de clusters SLIC.')
    parser.add_argument('--out_path_image2', type=str, default="Results/masks+slic/mask.png",
                        help='Chemin vers le fichier de sortie pour le masque en format image.')
    
    args = parser.parse_args()
    main(args)