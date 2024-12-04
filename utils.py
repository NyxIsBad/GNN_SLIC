import torch
import numpy as np
import cv2
import os
import torchvision.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import ToSLIC, RadiusGraph

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
def superpixels_to_2d_image(rec: Data, bg, scale: int = 30, edge_width: int = 1, size_tuple: tuple = (224,224)) -> np.ndarray:
    # Scale the positions of the superpixels
    rect_expand = scale * 5
    pos = (rec.pos.clone() * scale).int()
    new_size = (size_tuple[0] * scale, size_tuple[1] * scale)
    # Create a blank image to display the superpixels
    if bg is not None:
        bg_resized = cv2.resize(bg, new_size, interpolation=cv2.INTER_LINEAR)
        image = bg_resized.copy()
    else:
        image = np.full((*new_size, 3), fill_value=255, dtype=np.uint8) 
    
    # Draw superpixels as rectangles on the image
    for (color, (x, y)) in zip(rec.x, pos):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 - rect_expand//2, y0 - rect_expand//2
        x0, y0 = x0 + rect_expand//2, y0 + rect_expand//2
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x0 >= new_size[0]:
            x0 = new_size[0] - 1
        if y0 >= new_size[1]:
            y0 = new_size[1] - 1
        
        # Ensure color is a tensor with 3 elements (RGB)
        color = color.numpy()  # Convert tensor to NumPy array
        color = (color * 255).round().astype(np.uint8)
        color = tuple(color.tolist())  # Convert to RGB tuple
        # print(color, x0, y0, x1, y1)
        # Draw the rectangle with the adjusted color
        cv2.rectangle(
            img = image,
            pt1 = (x0, y0),
            pt2 = (x1, y1),
            color = color,
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

        cv2.line(image, (x0, y0), (x1, y1), color=(125,125,125), thickness=edge_width)

    return image

def slicify(image: np.ndarray, n_segments: int = 50, compactness: int = 10, radius: float = 10.0, max_neighbors: int = 32) -> Data:
    transform = T.Compose(
        [
            T.ToTensor(), 
            ToSLIC(n_segments=n_segments, compactness=compactness),
            RadiusGraph(r=radius, max_num_neighbors=max_neighbors)
        ]
    )
    data = transform(image)
    return data