import numpy as np
import cv2
import json
#import satlaspretrain_models
import torch
from scipy.signal import convolve2d
from scipy import ndimage
from PIL import Image


class PostProcessing:
    def __init__(self, predictions):
        self.predictions = predictions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def post_process_optimized(self, raw_output, patch_size):
        """Post-processes the raw output from the model using convolution for faster accumulation,
        and then determines class labels using argmax."""

        if not isinstance(raw_output, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if raw_output.shape != (1024, 1024, 5):
            raise ValueError("Input must have shape (1024, 1024, 5).")

        kernel = np.ones((patch_size, patch_size), dtype=np.float32).reshape(patch_size, patch_size, 1)
        # Use ndimage.convolve once for all channels
        accumulated_probs = ndimage.convolve(raw_output, kernel, mode='constant', cval=0)

        class_labels = np.argmax(accumulated_probs, axis=-1)
        return class_labels
    
    def post_process_torch(self, outputs, gamma=0.5):
        empty_img = torch.zeros((1024, 1024, 5), device=self.device)
        
        # Convert outputs to tensor once
        coords = torch.tensor([[x,y] for x,y,_ in outputs], device=self.device)
        probs = torch.tensor([p for _,_,p in outputs], device=self.device)
        
        # Batch assignment
        for coord, prob in zip(coords, probs):
            empty_img[coord[0]:coord[0]+64, coord[1]:coord[1]+64] += prob
        
        padded = torch.nn.functional.pad(empty_img, (0,0,1,1,1,1))
        neighbors = gamma * (padded[:-2, 1:-1] + padded[2:, 1:-1] + 
                            padded[1:-1, :-2] + padded[1:-1, 2:])
        
        return torch.argmax(empty_img + neighbors, dim=2)  # Keep on GPU if needed
    
    def converter(tensor):
    # Dictionary to store polygons for each unique value
        polygons = {}

        for val in range(5):  # Iterate over unique values (0-4)
            mask = (tensor == val).astype(np.uint8)  # Create binary mask for the value

            # Find contours (polygons) of connected components
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Convert contours to a list of polygons
            polygons[val] = [c.reshape(-1, 2).tolist() for c in contours]
        return polygons


    def write_json(polygons, filename:str):
    #Function that takes list of polygons and writes them to a json file.
    #The list of polygons is a list of lists of polygons, where each polygon is a list of points.
    # Dictionary to map unique values to class names
        numper_2_class={1:"plantation",2:"grassland_shrubland",3:"mining",4:"logging"}
        with open(f"{filename}.json", "w") as file:
            images_overview={"images":[]}
            # Iterate over images in polygons
            for i in range(len(polygons)):

                curr={"file_name":f"evaluation_{i}.tif","annotations":[]}
                #Iterate over types of polygons in image
                for j in range(1,len(polygons[i])):
                    # Convert polygons to list of coordinates

                    for k in range(len(polygons[i][j])):
                        listed_polygons=[]

                        for p in range(len(polygons[i][j][k])):
                            listed_polygons.append(polygons[i][j][k][p][0])
                            listed_polygons.append(polygons[i][j][k][p][1])
                        curr["annotations"].append({"class":numper_2_class[j],"segmentation":listed_polygons})
                images_overview["images"].append(curr)
            file.write(json.dumps(images_overview, ensure_ascii=False, indent=4))
    

    def display_prediction(self, assigned_labels):
        if isinstance(assigned_labels, torch.Tensor):
            assigned_labels = assigned_labels.cpu().numpy()  # Move to CPU if it's on GPU

        # Ensure y_test is of integer type
        assigned_labels = assigned_labels.astype(np.uint8)

        # Create a color mapping for the classes
        color_map = {
            0: [0, 0, 0],      # background: black
            1: [255, 0, 0],    # plantation: red
            2: [0, 255, 0],    # grassland_shrubland: green
            3: [0, 0, 255],    # mining: blue
            4: [255, 255, 0]     # logging: yellow
        }

        # Create an RGB image where each class is represented by its color
        height, width = assigned_labels.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                class_id = assigned_labels[i, j]
                rgb_image[i, j] = color_map[class_id]

        # Create a PIL image from the numpy array
        img = Image.fromarray(rgb_image)

        # Save the image to a file
        img.save("rasterized_prediction.png")

        print("Rasterized image saved to rasterized_prediction.png")