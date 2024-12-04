import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from skimage.segmentation import slic
from skimage import color
from skimage.util import img_as_float
import cv2
import PIL
from torch_geometric.data import Data

from utils import *

# ------------------- SLIC ------------------- #
IMAGE_DIR = './'
SIZE = (224, 224)
N_SEGMENTS = 500
COMPACTNESS = 1

print('Loading data...')

# Load images from the directory
images = load_dir(IMAGE_DIR)  # Replace with the path to your image directory
images = [cv2.resize(image, SIZE) for image in images]

sample_image = images[0]

slicified = slicify(sample_image, n_segments=N_SEGMENTS, compactness=COMPACTNESS, radius=SIZE[0]//8, max_neighbors=32)

print(slicified)

slicified_image = superpixels_to_2d_image(rec=slicified, scale=10, edge_width=5, size_tuple=SIZE, bg=sample_image)
slicified_graph = superpixels_to_2d_image(rec=slicified, scale=10, edge_width=5, size_tuple=SIZE, bg=None)

# plot the 3 images
plt.figure(figsize=(15, 15))
plt.subplot(131)
plt.imshow(sample_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(slicified_image)
plt.title('Superpixels')
plt.axis('off')

plt.subplot(133)
plt.imshow(slicified_graph)
plt.title('Graph Connectivity')
plt.axis('off')

plt.show()