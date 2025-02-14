import numpy as np
import cv2
import json
import torch
from scipy.signal import convolve2d
from scipy import ndimage
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch


class TestDataset(Dataset):
    """Dataset class for test data without labels"""
    def __init__(
        self,
        images: np.ndarray,  # Shape: (N, C, H, W)
        patch_size: int = 256,
        stride: int = 256
    ):
        self.images = images
        self.patch_size = patch_size
        self.stride = stride
        self.patches, self.positions = self._create_patches()

    def _create_patches(self):
        """Create patches and store their original positions"""
        patches = []
        positions = []  # Store (image_idx, y, x) for each patch
        N, C, H, W = self.images.shape

        for img_idx in range(N):
            image = self.images[img_idx]
            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    img_patch = image[:, y:y + self.patch_size, x:x + self.patch_size]
                    patches.append(img_patch)
                    positions.append((img_idx, y, x))

        return patches, positions

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return torch.as_tensor(self.patches[idx], dtype=torch.float32)

class PostProcessing:
    def __init__(self, model, test_data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.test_dataset = TestDataset(test_data, patch_size=256)
        self.test_dataloader = DataLoader(
                                        self.test_dataset, 
                                        batch_size=32, 
                                        shuffle=False, 
                                        num_workers=0,
    )
        
    

    def predict_probs(self):
        """
        Predicts the probabilities of each class for each pixel in the image.

        Args:
            model (torch.nn.Module): The trained model.
            test_dataloader (torch.utils.data.DataLoader): The DataLoader for the test dataset.

        Returns:
            list: A list of predictions. Each element contains 64 patches of predictions,
                with each patch containing 128x128 probabilities as tensors for each class.
        """
        print("Starting prediction...")
        all_predictions = []
        with torch.no_grad():
            self.model.eval()
            predictions = []
            for data in self.test_dataloader:
                data = data.to(self.device)

                output = self.model(data)[0]  # Assuming model returns a tuple
                probs = torch.nn.functional.softmax(output, dim=1)  # Keep as tensor

                # Iterate through the batch
                batch_predictions = []
                for batch_idx in range(data.shape[0]):
                    # Get coordinates of each pixel in the image
                    height, width = data.shape[2], data.shape[3]
                    x_coords = torch.arange(height, device=self.device)
                    y_coords = torch.arange(width, device=self.device)
                    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
                    x_coords = x_grid.flatten()
                    y_coords = y_grid.flatten()

                    # Combine coordinates and probabilities
                    image_predictions = []
                    for i in range(len(x_coords)):
                        x = x_coords[i].item()
                        y = y_coords[i].item()
                        pixel_probs = probs[batch_idx, :, x, y]  # Keep as tensor
                        image_predictions.append((x, y, pixel_probs))
                    batch_predictions.extend(image_predictions)
                predictions.append(batch_predictions)

            all_predictions.append(predictions)
        return all_predictions


    def reconstruct_from_patches(self, all_predictions, original_size=(1024, 1024), patch_size=128):
        """
        Reconstructs a full image from patches with probability lists.

        Args:
            predictions (list): A list of lists of lists of tuples. The outer list corresponds to images,
                                the middle list corresponds to batches, and the inner list contains tuples
                                of (x-coordinate, y-coordinate, probability list).
            original_size (tuple): The size of the original image (height, width). Default is (1024, 1024).
            patch_size (int): The size of the patches. Default is 128.

        Returns:
            list[torch.Tensor]: A list of 3D tensors representing the reconstructed images with shape (1024, 1024, 5).
                        Each pixel contains a probability distribution over the 5 classes.
        """
        # Initialize an empty list to hold the reconstructed images
        all_images = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for predictions in all_predictions:
            # Initialize tensor instead of numpy array
            full_img = torch.zeros((original_size[0], original_size[1], 5), device=device)

            # Iterate through the images
            for image_idx, image_predictions in enumerate(predictions):
                # Iterate through the batches
                for batch_idx, batch_patches in enumerate(image_predictions):
                    # Iterate through the patches and place them in the full image
                    for patch_data in batch_patches:
                        x, y, probs = patch_data  # Unpack the tuple

                        # Calculate the patch indices
                        patch_x_idx = (image_idx * len(image_predictions) + batch_idx) // (original_size[0] // patch_size)
                        patch_y_idx = (image_idx * len(image_predictions) + batch_idx) % (original_size[1] // patch_size)

                        # Calculate the actual pixel coordinates in the full image
                        pixel_x = x + patch_x_idx * patch_size
                        pixel_y = y + patch_y_idx * patch_size

                        # Convert probs to tensor if it's not already
                        probs_tensor = torch.tensor(probs, device=device) if not isinstance(probs, torch.Tensor) else probs
                        # Place the probability tensor into the corresponding pixel in the full image
                        full_img[pixel_x, pixel_y, :] = probs_tensor
            all_images.append(full_img)
        return all_images


    
    def post_process_torch(self, outputs, gamma=0.5):
        all_images = []
        for output in outputs:
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
            
            all_images.append(torch.argmax(empty_img + neighbors, dim=2))
        return all_images
    
    def converter(self, tensors):
        """
        Convert multiple image tensors to polygons.
        
        Args:
            tensors: List of tensors or single tensor. Each tensor should be (1024, 1024) 
                    containing class labels 0-4
        
        Returns:
            list[dict]: List of dictionaries, one per image, where each dictionary
                    contains polygons for each class {0: [...], 1: [...], ...}
        """
        # Handle single tensor case
        if isinstance(tensors, (np.ndarray, torch.Tensor)):
            tensors = [tensors]
        
        all_image_polygons = []
        
        for idx, tensor in enumerate(tensors):
            print("Converting tensor to polygons for image", idx)
            # Dictionary to store polygons for current image
            image_polygons = {}
            
            for val in range(5):  # Iterate over unique values (0-4)
                mask = (tensor == val).astype(np.uint8)  # Create binary mask for the value
                
                # Find contours (polygons) of connected components
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Convert contours to a list of polygons
                image_polygons[val] = [c.reshape(-1, 2).tolist() for c in contours]
            
            all_image_polygons.append(image_polygons)
        
        return all_image_polygons

    def post_process(self, gamma=0.5):
        predictions = self.predict_probs()
        reconstructed_images = self.reconstruct_from_patches(predictions)
        assigned_labels = self.post_process_torch(reconstructed_images, gamma=gamma)
        polygons = self.converter(assigned_labels)
        return polygons

    def write_json(self, polygons, filename:str):
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
    

    def display_prediction(self, assigned_labels, true_labels):
        """
        Display prediction and ground truth side by side.
        
        Args:
            assigned_labels: Model predictions (tensor or numpy array)
            true_labels: Ground truth labels (tensor or numpy array)
        """
        # Convert tensors to numpy if needed
        if isinstance(assigned_labels, torch.Tensor):
            assigned_labels = assigned_labels.cpu().numpy()
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy()

        # Ensure labels are of integer type
        assigned_labels = assigned_labels.astype(np.uint8)
        true_labels = true_labels.astype(np.uint8)

        # Create a color mapping for the classes
        color_map = {
            0: [0, 0, 0],      # background: black
            1: [255, 0, 0],    # plantation: red
            2: [0, 255, 0],    # grassland_shrubland: green
            3: [0, 0, 255],    # mining: blue
            4: [255, 255, 0]   # logging: yellow
        }

        # Create RGB images for both predictions and ground truth
        height, width = assigned_labels.shape
        pred_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        true_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        # Color the images
        for i in range(height):
            for j in range(width):
                pred_class = assigned_labels[i, j]
                true_class = true_labels[i, j]
                pred_rgb[i, j] = color_map[pred_class]
                true_rgb[i, j] = color_map[true_class]

        # Create a combined image with prediction and ground truth side by side
        combined_img = Image.new('RGB', (width * 2, height))
        combined_img.paste(Image.fromarray(pred_rgb), (0, 0))
        combined_img.paste(Image.fromarray(true_rgb), (width, 0))

        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        draw.text((width//4, 20), "Prediction", fill=(255,255,255), font=font, anchor="mm")
        draw.text((width + width//4, 20), "Ground Truth", fill=(255,255,255), font=font, anchor="mm")

        # Save the combined image
        combined_img.save("comparison_prediction.png")
        print("Comparison image saved to comparison_prediction.png")