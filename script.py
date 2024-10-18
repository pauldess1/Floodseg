import os
import shutil

# Directory containing the existing output files
output_dir = "Semantic_seg/Exemples/masks+slic"
# Directory containing the input images
image_dir = "Semantic_seg/FloodNet_images/queries"
# Destination directory for copied files
dest_dir = "Semantic_seg/Exemples/queries"

# Create the destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Iterate over each existing output file
for output_file in os.listdir(output_dir):
    if output_file.endswith("_mask_slic.png"):
        # Extract the identifier xxxxx
        basename = output_file[:-14]  # Remove "_mask_slic.png"
        identifier = basename.split('_')[1]
        
        # Construct the path to the corresponding input image
        img_path = os.path.join(image_dir, f"query_{identifier}.png")
        
        # Check if the input image exists
        if os.path.isfile(img_path):
            # Copy the input image to the destination directory
            shutil.copy(img_path, dest_dir)
        else:
            print(f"Input image {img_path} not found, skipped.")
