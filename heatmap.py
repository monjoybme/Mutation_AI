import os
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
from openslide import OpenSlide
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from model import build_custom_resnet50  # Use the custom ResNet50 model from model.py
from tqdm import tqdm  # Progress bar
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 9331200000
from datetime import datetime
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor

# Set paths
wsi_dir = '/Path/to/the/WSIs/'
Save_results_path = '/Save/path/'
trained_model_path = '/Path/to/the/trained_model/'

columns = ['EGFR', 'KRAS', 'TP53', 'RBM10', 'EGFR_pL591R', 'EGFR_pE479_A483del', 'KRAS_pG12C',
           'KRAS_pG12V', 'KRAS_pG12D', 'CDKN2A_deletion', 'MDM2_amplification', 'ALK_fusion',
           'WGD', 'Kataegis', 'APOBEC', 'TMB']

num_classes = len(columns)
patch_size = 256
target_size = (256, 256)

# Function to identify tissue regions
def identify_tissue_regions(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary

# Function to extract patches from identified tissue regions
def extract_patches_from_tissue(img, tissue_contours, patch_size):
    patches = []
    for contour in tissue_contours:
        x, y, w, h = cv2.boundingRect(contour)
        for y_patch in range(y, y + h - patch_size + 1, patch_size):
            for x_patch in range(x, x + w - patch_size + 1, patch_size):
                patch = img[y_patch:y_patch + patch_size, x_patch:x_patch + patch_size, :]
                patches.append((patch, (x_patch, y_patch)))
    return patches

# Function to read whole slide image
def read_wsi(fname, level=0):
    slide = OpenSlide(fname)
    # Check the available levels
    print(len(slide.level_dimensions))
    if level >= len(slide.level_dimensions):
        level = len(slide.level_dimensions) - 1  # Use the maximum available level
    slide_region = slide.read_region((0, 0), level, slide.level_dimensions[level])
    img = np.array(slide_region.convert('RGB'))
    slide.close()  # Close the slide after reading
    return img

# Function to calculate patch-wise probabilities
def calculate_patch_probabilities(patch, model):
    patch_resized = cv2.resize(patch, target_size)  # Resize to match model's input size
    x = image.img_to_array(patch_resized) / 255.  # Normalize
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    probabilities = model.predict(x)[0]
    return probabilities

# Build and load the custom ResNet model
model = build_custom_resnet50((*target_size, 3), num_classes)
model.load_weights(trained_model_path + 'custom_resnet50.hdf5')  # Load the weights

# Function to process a single WSI
def process_wsi(single_image):
    single_image_path = os.path.join(wsi_dir, single_image)
    
    # Read WSI and extract patches from identified tissue regions
    img = read_wsi(single_image_path, level=0)
    base = os.path.basename(single_image_path)
    filename = os.path.splitext(base)[0]
    tissue_contours, binary_mask = identify_tissue_regions(img)
    patches = extract_patches_from_tissue(img, tissue_contours, patch_size)
    
    # Initialize a blank canvas for the probability map with the same size as the WSI
    prob_map = np.zeros((img.shape[0], img.shape[1], num_classes), dtype=np.float32)
    count_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    
    # Calculate probabilities for each patch
    for patch, (x, y) in patches:
        probabilities = calculate_patch_probabilities(patch, model)
        
        # Store the probabilities in the corresponding location on the probability map
        prob_map[y:y + patch_size, x:x + patch_size] += probabilities
        count_map[y:y + patch_size, x:x + patch_size] += 1
    
    # Normalize the probability map by dividing by the count map
    prob_map /= np.expand_dims(count_map, axis=-1)
    prob_map[np.isnan(prob_map)] = 0  # Replace NaNs with 0

    # Apply Gaussian filter to smooth the probability map
    prob_map_smoothed = gaussian_filter(prob_map, sigma=(patch_size//2, patch_size//2, 0))

    # Generate and save heatmaps for each class
    for class_idx in range(num_classes):
        heatmap = prob_map_smoothed[:, :, class_idx]
        
        # Normalize heatmap to the range 0-255
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = np.uint8(heatmap_norm)
        
        # Apply colormap
        heatmap_colormap_pre = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Convert BGR to RGB (Matplotlib uses RGB)
        heatmap_colormap_rgb = cv2.cvtColor(heatmap_colormap_pre, cv2.COLOR_BGR2RGB)

        # Enhance contrast (example: simple gamma correction)
        gamma = 1.5
        heatmap_colormap = np.clip((heatmap_colormap_rgb / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
        
        # Create a mask where the tissue regions are True
        tissue_mask = binary_mask == 255
        
        # Superimpose heatmap on the tissue regions only
        heatmap_combined = img.copy()
        heatmap_combined[tissue_mask] = cv2.addWeighted(heatmap_colormap[tissue_mask], 0.5, img[tissue_mask], 0.5, 0)
        
        # Save the final heatmap image using cv2.imwrite
        outfile = f'{Save_results_path}/{filename}_heatmap_{columns[class_idx]}_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
        cv2.imwrite(outfile, cv2.cvtColor(heatmap_combined, cv2.COLOR_RGB2BGR))

# Process each WSI in the directory with a progress bar
# Assuming wsi_dir is the directory containing WSI files
wsi_files = [f for f in os.listdir(wsi_dir) if f.endswith(('.svs', '.ndpi', '.scn'))]

# Using ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:
    list(tqdm(executor.map(process_wsi, wsi_files), total=len(wsi_files), desc="Processing WSIs"))
