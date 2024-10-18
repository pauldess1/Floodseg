import os
import subprocess

# Path to the directory containing images
image_dir = "VPR/Datasets/vpair_complet/Queries/queries"
# Path to the PSPNet model
model_path = "segmentation/weights/pspnet_best.pth"
# Output directories for the results
out_array_dir = "Results/arrays"
out_image1_dir = "Results/masks"
out_image2_dir = "Results/masks+slic"
slic_contours_dir = "Results/arrays"
slic_clusters_dir = "Results/arrays"

# Create output directories if they don't exist
os.makedirs(out_array_dir, exist_ok=True)
os.makedirs(out_image1_dir, exist_ok=True)
os.makedirs(out_image2_dir, exist_ok=True)
os.makedirs(slic_contours_dir, exist_ok=True)
os.makedirs(slic_clusters_dir, exist_ok=True)

# Iterate through each image in the directory
for img_name in os.listdir(image_dir):
    if img_name.endswith(".png") and img_name.startswith("query_"):
        img_path = os.path.join(image_dir, img_name)
        basename = os.path.splitext(img_name)[0]
        
        # Define output paths
        out_path_array = os.path.join(out_array_dir, f"{basename}_mask.npy")
        out_path_image1 = os.path.join(out_image1_dir, f"{basename}_mask.png")
        slic_contours_path = os.path.join(slic_contours_dir, f"{basename}_contours.npy")
        slic_clusters_path = os.path.join(slic_clusters_dir, f"{basename}_clusters.npy")
        out_path_image2 = os.path.join(out_image2_dir, f"{basename}_mask_slic.png")

        # Run the segmentation script
        subprocess.run([
            "python", "run.py",
            "--model_path", model_path,
            "--img_path", img_path,
            "--out_path_array", out_path_array,
            "--out_path_image1", out_path_image1,
            "--slic_contours_path", slic_contours_path,
            "--slic_clusters_path", slic_clusters_path,
            "--out_path_image2", out_path_image2
        ])
