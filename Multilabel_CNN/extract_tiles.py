"""
python extract_tiles.py -i /path/to/your.wsi -o /path/to/output/directory/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import logging
import math
import numpy as np
import os
import openslide
from PIL import Image
from skimage.color import rgb2gray
from functions_monjoy import save_image  # Assuming this function is defined elsewhere as per your provided code

# Set patch size to 256 for tile extraction
PATCH_SIZE = 256

parser = argparse.ArgumentParser(description='Extract a series of patches from a whole slide image')
parser.add_argument("-i", "--image", dest='wsi',  nargs='+', required=True, help="path to a whole slide image")
parser.add_argument("-o", "--output", dest='output_name', default="output", help="Output directory [default: `output/`]")

parser.add_argument("-v", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="INFO",
                    help="Set the logging level")
args = parser.parse_args()

if args.logLevel:
    logging.basicConfig(level=getattr(logging, args.logLevel))

wsi = ' '.join(args.wsi)

# Set global variables
mean_grey_values = 0.7 * 255  # Grey scale threshold for tissue
number_of_useful_regions = 0
wsi = os.path.abspath(wsi)
outname = os.path.abspath(args.output_name)
basename = os.path.basename(wsi)
level = 0

def main():
    img, num_x_patches, num_y_patches = open_slide()

    logging.debug(f'img: {img}, num_x_patches = {num_x_patches}, num_y_patches: {num_y_patches}')
    
    for x in range(num_x_patches):
        for y in range(num_y_patches):
            img_data = img.read_region((x * PATCH_SIZE, y * PATCH_SIZE), level, (PATCH_SIZE, PATCH_SIZE))
            save_patch(x * PATCH_SIZE, y * PATCH_SIZE, img_data, img)

    pc_uninformative = number_of_useful_regions / (num_x_patches * num_y_patches) * 100
    pc_uninformative = round(pc_uninformative, 2)
    logging.info(f'Completed patch extraction of {number_of_useful_regions} images.')
    logging.info(f'{pc_uninformative}% of the image is uninformative\n')

def save_patch(x_top_left, y_top_left, img_data, img):
    """ Save patch as image after checking if it's informative. """
    img_data_np = np.array(img_data)
    grey_img = rgb2gray(img_data_np)
    
    logging.debug(f'Image grayscale = {np.mean(grey_img)} compared to threshold {mean_grey_values}')
    
    # If tissue is detected (based on grey scale threshold)
    if np.mean(grey_img) < mean_grey_values:
        global number_of_useful_regions
        number_of_useful_regions += 1
        wsi_base = os.path.basename(wsi).split('.')[0]
        img_name = f'{wsi_base}_{x_top_left}_{y_top_left}_{PATCH_SIZE}'
        logging.debug(f'Saving patch: {x_top_left}, {y_top_left}, {np.mean(grey_img)}')
        save_image(img_data_np, 1, img_name)

def open_slide():
    """ Open the slide and calculate number of patches. """
    logging.debug(f'Opening slide: {wsi}')

    img = openslide.OpenSlide(wsi)
    img_dim = img.level_dimensions[0]
    logging.debug(f'Image dimensions: {img_dim}')

    # Calculate number of patches based on the new patch size
    num_x_patches = img_dim[0] // PATCH_SIZE
    num_y_patches = img_dim[1] // PATCH_SIZE

    logging.debug(f'Calculated {num_x_patches} x-patches and {num_y_patches} y-patches.')
    return img, num_x_patches, num_y_patches

def validate_dir_exists():
    """ Ensure the output directory exists. """
    if not os.path.isdir(outname):
        os.mkdir(outname)
    logging.debug(f'Validated directory {outname} exists.')

if __name__ == '__main__':
    validate_dir_exists()
    main()
