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
        
    

    def predict_probs(self, model, test_dataloader):
        model.eval()
        all_probs = []
        with torch.no_grad():
            for data in test_dataloader:
                data = data.to(self.device)
                output = model(data)[0]  # Shape: (batch_size, 5, 128, 128)
                probs = torch.nn.functional.softmax(output, dim=1)
                # Permute to (batch_size, 128, 128, 5) and collect
                probs = probs.permute(0, 2, 3, 1).cpu().numpy()
                all_probs.append(probs)
        # Concatenate all batches into (num_patches, 128, 128, 5)
        return np.concatenate(all_probs, axis=0)


    def stitch_patches(self, prob_patches, positions, image_shape=(1024, 1024), patch_size=128):
        num_classes = prob_patches.shape[-1]
        stitched_images = []
        # Extract unique image indices from positions
        image_indices = sorted(set(pos[0] for pos in positions))
        for img_idx in image_indices:
            full_image = np.zeros((image_shape[0], image_shape[1], num_classes))
            # Iterate through all patches and place them if they belong to the current image
            for i, (curr_idx, y, x) in enumerate(positions):
                if curr_idx == img_idx:
                    full_image[y:y+patch_size, x:x+patch_size, :] = prob_patches[i]
            stitched_images.append(full_image)
        return [torch.tensor(img) for img in stitched_images]



    
    def post_process_torch(self, outputs, gamma=0.5):
        all_images = []
        for img in outputs:  # Each img is a tensor of shape (1024, 1024, 5)
            # Pad the image to handle borders
            padded = torch.nn.functional.pad(img, (0, 0, 1, 1, 1, 1))  # Pad H and W by 1
            
            # Compute contributions from 4-directional neighbors
            top = padded[:-2, 1:-1, :]    # Shape: (1024, 1024, 5)
            bottom = padded[2:, 1:-1, :]
            left = padded[1:-1, :-2, :]
            right = padded[1:-1, 2:, :]
            
            # Combine neighbor contributions and add to original probabilities
            neighbors = gamma * (top + bottom + left + right)
            combined_probs = img + neighbors
            
            # Get final predictions
            class_predictions = torch.argmax(combined_probs, dim=2)
            all_images.append(class_predictions)
        
        return all_images  # List of (1024, 1024) tensors with class indices
    
    def converter(self, tensors):
        """
        Convert multiple image tensors to polygons.
        
        Args:
            tensors: List of tensors or single tensor. Each tensor should be (1024, 1024) 
                    containing class labels 0-4 (either numpy array or torch.Tensor)
        
        Returns:
            list[dict]: List of dictionaries, one per image, where each dictionary
                    contains polygons for each class {0: [...], 1: [...], ...}
        """
        # Handle single tensor case and convert to numpy
        if isinstance(tensors, (np.ndarray, torch.Tensor)):
            tensors = [tensors]
        
        all_image_polygons = []
        
        for idx, tensor in enumerate(tensors):
            print("Converting tensor to polygons for image", idx)
            image_polygons = {}
            
            # Convert to numpy and ensure proper 2D format
            if isinstance(tensor, torch.Tensor):
                # Handle both CPU and GPU tensors
                tensor_np = tensor.cpu().detach().numpy().squeeze().astype(np.uint8)
                print(tensor_np.shape)
            else:
                tensor_np = tensor.squeeze().astype(np.uint8)
            
            # Critical validation
            if tensor_np.ndim != 2:
                raise ValueError(f"Input tensor must be 2D after squeezing. Got {tensor_np.shape}")
            
            for class_id in range(5):
                # Create binary mask
                mask = (tensor_np == class_id).astype(np.uint8)
                
                # Verify OpenCV requirements
                if not isinstance(mask, np.ndarray):
                    raise TypeError(f"Mask must be numpy array, got {type(mask)}")
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
                
                # Find contours
                contours, _ = cv2.findContours(
                    mask, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Convert and filter contours
                image_polygons[class_id] = [
                    contour.squeeze().tolist() 
                    for contour in contours 
                    if contour.shape[0] >= 3  # Minimum 3 points for polygon
                ]
            
            all_image_polygons.append(image_polygons)
        
        return all_image_polygons

    def post_process(self, gamma=0.5):
        predictions = self.predict_probs(self.model, self.test_dataloader)
        reconstructed_images = self.stitch_patches(predictions)
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