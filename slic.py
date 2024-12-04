import torch
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

print('Loading data...')

# Load images from the directory
images = load_dir('./')  # Replace with the path to your image directory
# resize to 224x224
images = [cv2.resize(image, (224, 224)) for image in images]
sample_image = images[0]

# show the image in PIL
PIL.Image.fromarray(sample_image).show()

graph = slicify(sample_image, n_segments=50, compactness=10)

# print graph
print(graph)

# plot graph
plt.imshow(sample_image)
# Show graph boundaries

plt.axis('off')
plt.show()