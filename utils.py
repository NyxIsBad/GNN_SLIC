import numpy as np
import cv2
import os
import torchvision.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import ToSLIC as ToSLICGeometric

# Function to load images from the directory
def load_dir(path: str) -> np.ndarray:
    # Get all *.png, *.jpg, *.jpeg files in the directory
    files = [f for f in os.listdir(path) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
    images = []
    for file in files:
        image = cv2.imread(os.path.join(path, file))
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return np.array(images)

# Function to convert superpixels to a 2D image using torch_geometric Data format
def superpixels_to_2d_image(rec: Data, scale: int = 30, edge_width: int = 1) -> np.ndarray:
    # Scale the positions of the superpixels
    pos = (rec.pos.clone() * scale).int()

    # Create a blank image to display the superpixels
    image = np.zeros((scale * 224, scale * 224, 3), dtype=np.uint8)  # RGB image
    
    # Draw superpixels as rectangles on the image
    for (color, (x, y)) in zip(rec.x, pos):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 - scale, y0 - scale
        
        # Ensure color is a tensor with 3 elements (RGB)
        color = color.clone().detach().cpu().numpy()  # Convert tensor to NumPy array
        color = np.clip(color + 0.15, 0, 1) * 255  # Adjust intensity (clip to valid range)
        color = color.astype(int)  # Convert to integer values
        
        # Draw the rectangle with the adjusted color
        cv2.rectangle(
            img = image,
            pt1 = (x0, y0),
            pt2 = (x1, y1),
            color = tuple(color.tolist()), # takes BGR (tuple of 3 integers)
            thickness = -1  # Fill the rectangle
        )

    # Draw edges (graph connectivity)
    for node_ix_0, node_ix_1 in rec.edge_index.T:
        x0, y0 = list(map(int, pos[node_ix_0]))
        x1, y1 = list(map(int, pos[node_ix_1]))

        x0 -= scale // 2
        y0 -= scale // 2
        x1 -= scale // 2
        y1 -= scale // 2

        cv2.line(image, (x0, y0), (x1, y1), 125, edge_width)

    return image


def slicify(image: np.ndarray, n_segments: int = 50, compactness: int = 10) -> Data:
    transform = T.Compose([T.ToTensor(), ToSLICGeometric(n_segments=n_segments, compactness=compactness)])
    data = transform(image)
    return data