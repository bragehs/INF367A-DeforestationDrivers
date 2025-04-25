import os
import json
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import cv2
from PIL import Image, ImageDraw
from natsort import natsorted
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
import torch
from typing import List, Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from tqdm import tqdm
from skimage.exposure import match_histograms
from skimage.morphology import remove_small_objects
from scipy import ndimage

class PreProcessing:
    # structure: {class name: (color, RGB, class label)
    annotation_color_allocation = {
    'background' : ('black', (0, 0, 0), 0), 
    'plantation' : ('red', (255, 0, 0), 1), 
    'grassland_shrubland' : ('green', (0, 255, 0), 2), 
    'mining' : ('blue', (0, 0, 255), 3), 
    'logging' : ('yellow', (255, 255, 0), 4) 
}   
    def __init__(self, train_set=True, satlas = True, img_labels = False):
        self.base_dir = os.getcwd()
        self.parent_dir = os.path.split(self.base_dir)[0]

        #images should be in parent_dir/train_images
        self.TRAIN_IMAGES_PATH=self.parent_dir+"/train_images"
        self.TEST_IMAGES_PATH=self.parent_dir+"/evaluation_images"
        self.img_labels_images_path =self.parent_dir+"/train-tif-v2"

        self.train_annotations = {}  # Initialize as empty dictionary
        self.polygons = {}

        #train_annotations does not have to be in parent dir as it is a small file
        if os.path.exists('train_annotations.json'): 
            with open('train_annotations.json', 'r') as file:
                self.train_annotations = json.load(file)
            for image in self.train_annotations["images"]:
                polys=[]
                for polygons in image["annotations"]:
                    geom = np.array(polygons['segmentation'])
                    polys.append((polygons["class"],geom))
                self.polygons[image["file_name"]]=polys

        self.train_set=train_set 
        self.satlas = satlas 
        self.img_labels = img_labels
        self.polygons = {}
        if train_set:
            self.train_images = [f for f in os.listdir(self.TRAIN_IMAGES_PATH) if f.endswith('.tif')]
        if img_labels:    
            self.img_labels_images = [f for f in os.listdir(self.img_labels_images_path) if f.endswith('.tif')]
        elif not train_set and not img_labels:
            self.test_images = [f for f in os.listdir(self.TEST_IMAGES_PATH) if f.endswith('.tif')]


    def convert_to_geojson(self, data):
        """
        Converts a list of dictionaries in the specified format to GeoJSON
        Args:
            data: A list of dictionaries containing 'class' and 'segmentation' keys
        Returns:
            A GeoJSON feature collection
        """
        features = []
        for item in data:
            polygon = []
            for i in range(0, len(item['segmentation']), 2):
                polygon.append([item['segmentation'][i], item['segmentation'][i+1]])
            features.append({
                "type": "Feature",
                "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
                },
                "properties": {"class": item['class']}
            })
        return { "type": "FeatureCollection", "features": features }
    
    def visualize_in_RGB(self, n=10, start=0, annotation=True, train = True):
        """
        Visualizes RGB images with optional annotations.
        """
        outfolder = 'rgb_annotation'  
        for index_for_image in range(start, n):
            filename = os.path.join(outfolder, f'rgb_with_annotation_{index_for_image}.png')
            if os.path.exists(filename):
                print(f"File {filename} already exists. Skipping.")
                continue
            # Visualize sample tif data
            if train:
                SAMPLE_TIF_PATH = f'{self.TRAIN_IMAGES_PATH}/train_{index_for_image}.tif' if self.train_set else f'{self.TEST_IMAGES_PATH}/evaluation_{index_for_image}.tif'
                annotation_data = self.train_annotations['images']
            else:
                SAMPLE_TIF_PATH = f'{self.TEST_IMAGES_PATH}/evaluation_{index_for_image}.tif'
                annotation_data = self.train_annotations['images']

            id_to_annotation = {item['file_name']: item for item in annotation_data}
            annotation_data = id_to_annotation[ f'train_{index_for_image}.tif']['annotations']
            
            # Convert to GeoJSON
            geojson_data = self.convert_to_geojson(annotation_data)
            gdf = gpd.GeoDataFrame.from_features(geojson_data)
            # Open sample tif file
            with rasterio.open(SAMPLE_TIF_PATH) as src:
                # Read bands 2, 3, and 4 (B, G, R)
                b2 = src.read(2)
                b3 = src.read(3)
                b4 = src.read(4)

                rgb_image = np.dstack((b4, b3, b2))

                rgb_image = np.nan_to_num(rgb_image) # Replace NaN and inf with 0

                # Normalize pixel values for display
                rgb_image = rgb_image.astype(np.float32)
                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
                rgb_image = np.clip(rgb_image, 0, 1) # Clip values to 0-1 range for floats

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(rgb_image)
                
                for idx in range(len(gdf)):
                    if annotation:
                        gdf.iloc[[idx]].boundary.plot(ax=ax, color=PreProcessing.annotation_color_allocation[gdf.iloc[idx]['class']][0])

                # Create and add custom legend
                legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                                            markerfacecolor=value[0], markersize=10) 
                                    for key, value in PreProcessing.annotation_color_allocation.items() 
                                    if key in gdf['class'].unique()]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

                # gdf.boundary.plot(ax=ax, color='red')
                plt.title(f'idx {index_for_image}: RGB Image')


    def visualize_rgb_by_class(self, class_label, max_images=10, show_overlay=False):
        """
        Visualizes all RGB images that contain the specified class.
        
        Args:
            class_label (str): Class to visualize (e.g., 'plantation', 'mining', etc.)
            max_images (int): Maximum number of images to display
            show_overlay (bool): Whether to show class regions with red overlay
        """
        # Always run rasterize to ensure we have data
        print("Generating rasterized masks...")
        self.rasterize()
        
        if class_label not in self.annotation_color_allocation:
            print(f"Class {class_label} not found in annotation_color_allocation.")
            return
        
        # Get the target color for the class
        target_color = self.annotation_color_allocation[class_label][1]
        
        # Find images that contain the specified class
        selected_images = []
        selected_keys = []
        
        print(f"Searching for images containing class '{class_label}'...")
        for key, rgb_array in self.RGB_raster_imgs.items():
            mask = ((rgb_array[:, :, 0] == target_color[0]) & 
                    (rgb_array[:, :, 1] == target_color[1]) & 
                    (rgb_array[:, :, 2] == target_color[2]))
            
            if np.any(mask):
                print(f"Found class in image: {key}")
                selected_keys.append(key)
                
                # Load the actual RGB image
                SAMPLE_TIF_PATH = f'{self.TRAIN_IMAGES_PATH}/{key}'
                with rasterio.open(SAMPLE_TIF_PATH) as src:
                    # Read bands 4, 3, and 2 (R, G, B)
                    r = src.read(4)
                    g = src.read(3)
                    b = src.read(2)
                    
                    # Stack bands for RGB image
                    rgb_img = np.dstack((r, g, b))
                    
                    # Normalize for display
                    rgb_img = np.nan_to_num(rgb_img)
                    rgb_img = rgb_img.astype(np.float32)
                    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                    rgb_img = np.clip(rgb_img, 0, 1)
                    
                    selected_images.append((rgb_img, mask))
                
                if len(selected_images) >= max_images:
                    break
        
        if not selected_images:
            print(f"No images found with class {class_label}.")
            return
        
        print(f"Displaying {len(selected_images)} images with class '{class_label}'...")
        
        num_images = len(selected_images)
        num_cols = min(3, num_images)
        num_rows = int(np.ceil(num_images / num_cols))
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))
        if num_images == 1:
            axes = np.array([axes])
        axes = np.array(axes).reshape(-1)
        
        for i, ((rgb_img, mask), key) in enumerate(zip(selected_images, selected_keys)):
            # Plot RGB image
            axes[i].imshow(rgb_img)
            
            if show_overlay:
                overlay = np.zeros_like(rgb_img)
                overlay[mask] = [1, 0, 0]  
                axes[i].imshow(overlay, alpha=0.3)
            
            axes[i].set_title(f"Image: {key}")
            axes[i].axis("off")
        
        # Turn off any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        
        plt.suptitle(f"RGB Images containing '{class_label}'", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    def analyze_nans(self, tif_path):
        """Analyze NaN distribution in TIF file"""
        with rasterio.open(tif_path) as src:
            data = src.read()
            meta = src.meta.copy()
            nan_mask = np.isnan(data)
            stats = {
                'total_pixels': data.size,
                'nan_count': np.count_nonzero(nan_mask),
                'nan_percentage': (np.count_nonzero(nan_mask)/data.size)*100
            }
        return data, meta, stats

    def fill_nans(self, data, method='open_cv_inpaint_telea', **kwargs):
        """Fill NaN values using specified method."""
        filled_data = np.zeros_like(data)
        
        for b in range(data.shape[0]):  # Iterate over bands
            band = data[b].copy()
            # Ensure the data is in the correct format for cv2.inpaint
            band = band.astype(np.float32)
            mask = np.isnan(band).astype(np.uint8)
            
            if method == 'open_cv_inpaint_telea' or method == 'open_cv_inpaint_ns':
                # Replace NaNs with 0 for inpainting
                band[mask == 1] = 0
                # Convert band to proper format for inpainting
                band_norm = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)
                band_uint8 = band_norm.astype(np.uint8)
                
                filled = cv2.inpaint(
                    band_uint8,
                    mask,  # Must be uint8
                    inpaintRadius=kwargs.get('inpaint_radius', 3),
                    flags=cv2.INPAINT_TELEA if method == 'open_cv_inpaint_telea' else cv2.INPAINT_NS
                )
                
                # Convert back to original range
                filled = filled.astype(np.float32)
                filled = cv2.normalize(filled, None, np.min(band[~np.isnan(band)]), np.max(band[~np.isnan(band)]), cv2.NORM_MINMAX)
                
            filled_data[b] = filled
        
        return filled_data
    
    def visualize_nan_filling(self, start=0, n=1, band_start=1, band_n=12, method= "open_cv_inpaint_telea"):
        """Visualize NaN filling results for a specific band."""
        for idx in range(start, n):
            for band in range(band_start, band_n +1):
                SAMPLE_TIF_PATH = f'{self.TRAIN_IMAGES_PATH}/train_{idx}.tif' if self.train_set else f'{self.TEST_IMAGES_PATH}/evaluation_{idx}.tif'
                with rasterio.open(SAMPLE_TIF_PATH) as src:
                    original = src.read(band)
    
                # Process the band
                filled = self.fill_nans(original[np.newaxis, ...], method=method)[0]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                im1 = ax1.imshow(original, cmap='viridis')
                ax1.set_title('Original with NaNs')
                plt.colorbar(im1, ax=ax1)
                
                im2 = ax2.imshow(filled, cmap='viridis')
                ax2.set_title(f'After {method} Filling')
                plt.colorbar(im2, ax=ax2)
                
                plt.suptitle(f'train_{idx}.tif - Band {band}')
                plt.tight_layout()
    
    def create_polygons(self, json_file='train_annotations.json'):
        '''
        Create polygons from the JSON file.
        '''
        with open(json_file, 'r') as file:
            input_annotations = json.load(file)

        polygons = {}

        for image in input_annotations["images"]:
            polys = []
            for annotation in image["annotations"]:
                geom = np.array(annotation['segmentation'])
                polys.append((annotation["class"], geom))
            polygons[image["file_name"]] = polys
        return polygons
    
    def rasterize(self, json_file='train_annotations.json'):
        '''
        Rasterize the polygons from the JSON file. 
        '''
        with open(json_file, 'r') as file:
            input_annotations = json.load(file)
        shape = (1024, 1024)
        self.RGB_raster_imgs={}
        polygons = self.create_polygons(json_file)
        
        for current_image in range(len(input_annotations['images'])):
            Image.MAX_IMAGE_PIXELS = None
            img = Image.new('RGB', (shape[1], shape[0]), (0, 0, 0))  # (w, h)

            for i in range(len(polygons[f"train_{current_image}.tif"])):

                poly = polygons[f"train_{current_image}.tif"][i][1]
                type_deforest= polygons[f"train_{current_image}.tif"][i][0]

                points = list(zip(poly[::2], poly[1::2]))
                points = [(x, y) for x, y in points]
                color = PreProcessing.annotation_color_allocation[type_deforest][1]
                ImageDraw.Draw(img).polygon(points, outline=None, fill=color)
            mask_2 = np.array(img)
            self.RGB_raster_imgs[f"train_{current_image}.tif"]=mask_2
    
    def visualize_rasterized(self, start=0, n=10, json_file='train_annotations.json'):
        '''
        Visualize the rasterized images.
        '''
        self.rasterize(json_file)
        for i in range(start, n):
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            plt.imshow(self.RGB_raster_imgs[f"train_{i}.tif"])

        # Add padding around the actual image
            plt.subplots_adjust(right=0.85)
            
            # Create legend text in whitespace
            legend_x = 1.02  # Position outside of image

            # Add legend items with consistent spacing
            for idx, (label, color) in enumerate(PreProcessing.annotation_color_allocation.items()):
                y_pos = 0.9 - (idx * 0.1)  # Start from top with even spacing
                plt.text(legend_x, y_pos, f'{label}', 
                        transform=ax.transAxes,
                        bbox=dict(facecolor=color[0], alpha=0.7, pad=5),
                        color='white' if color[0] in ['black', 'blue', 'red'] else 'black')

            plt.tight_layout()
    def visualize_rasterized_by_class(self, class_label, max_images=10):
        """
        Visualizes all rasterized images that contain the specified class.
        """
        if not hasattr(self, "RGB_raster_imgs") or not self.RGB_raster_imgs:
            print("Rasterized images not found. Running rasterize() first...")
            self.rasterize()
        
        if class_label not in PreProcessing.annotation_color_allocation:
            print(f"Class {class_label} not found in annotation_color_allocation.")
            return
        
        target_color = PreProcessing.annotation_color_allocation[class_label][1]
        
        selected_images = []
        selected_keys = []
        for key, rgb_array in self.RGB_raster_imgs.items():
            mask = ((rgb_array[:, :, 0] == target_color[0]) &
                    (rgb_array[:, :, 1] == target_color[1]) &
                    (rgb_array[:, :, 2] == target_color[2]))
            if np.any(mask):
                selected_images.append(rgb_array)
                selected_keys.append(key)
            if len(selected_images) >= max_images:
                break
        
        if not selected_images:
            print(f"No images found with class {class_label}.")
            return
        
        num_images = len(selected_images)
        num_cols = min(3, num_images)
        num_rows = int(np.ceil(num_images / num_cols))
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        axes = np.array(axes).reshape(-1)
        for i, (img, key) in enumerate(zip(selected_images, selected_keys)):
            axes[i].imshow(img)
            axes[i].set_title(f"Image: {key}\nContains: {class_label}")
            axes[i].axis("off")
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"Rasterized Images with Class: {class_label}", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def label_pixels(self):
        """Label pixels in the rasterized mask"""
        self.rasterize()
        self.labels = np.empty((len(self.RGB_raster_imgs), 1024, 1024), dtype=np.uint8)

        rgb_to_class = {
        tuple(v[1]): v[2]  # (R, G, B) -> class_label
        for k, v in PreProcessing.annotation_color_allocation.items()
    }

        for idx, rgb_array in enumerate(self.RGB_raster_imgs.values()):
            # Initialize label array with background class (0)
            label_array = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
            
            # Create masks for each class and assign labels
            for (r, g, b), class_id in rgb_to_class.items():
                if class_id == 0:  # Skip background (already initialized)
                    continue
                mask = (rgb_array[:, :, 0] == r) & \
                    (rgb_array[:, :, 1] == g) & \
                    (rgb_array[:, :, 2] == b)
                label_array[mask] = class_id
                
            self.labels[idx] = label_array    
    
    def normalize_sentinel2(self, tci_rgb, bands):
        # tci_rgb: shape (height, width, 3), 8-bit data in [0..255]
        tci_norm = (tci_rgb - tci_rgb.min()) / (tci_rgb.max() - tci_rgb.min())

        # bands: list of np arrays [B05, B06, B07, B08, B11, B12], each 16-bit
        normalized_bands = []
        for band in bands:
            norm_band = np.clip(band / 8160.0, 0, 1)
            normalized_bands.append(norm_band)

        # Stack into final tensor
        normalized_bands = np.array(normalized_bands)
        print(normalized_bands.shape, tci_norm.shape)
        return np.concatenate([tci_norm, normalized_bands], axis=1)

    def normalize_image(self, image):
        """Normalizes each channel of an image to the range [0, 1].

        Args:
            image: The input image.

        Returns:
            The normalized image.
        """

        # Normalize each channel independently
        num_channels = image.shape[0]  # Assuming image shape is (C, H, W)
        normalized_image = np.zeros_like(image, dtype=np.float32)

        for channel in range(num_channels):
            min_val = image[channel].min()
            max_val = image[channel].max()
            normalized_image[channel] = (image[channel] - min_val) / (max_val - min_val)

        return normalized_image   
    
    def verify_no_nans(self):
        """Verify that there are no NaN values in the prepared data and labels."""
        if np.isnan(self.prepared_data).any():
            raise ValueError("NaN values found in prepared_data!")
        if hasattr(self, 'labels') and np.isnan(self.labels).any():
            raise ValueError("NaN values found in labels!")
        print("Verification complete. No NaN values found.")

    def preprocess(self, method='open_cv_inpaint_telea'):
        """Process TIF file and save result"""
        if self.train_set:
            images = self.train_images
            self.prepared_data = np.zeros((len(images), 12, 1024, 1024))
        elif self.img_labels:
            df = pd.read_csv("filtered_balanced_dataset.csv")
            filtered_images = df['image_name'].tolist() if 'image_name' in df.columns else []
            filtered_images = [img + '.tif' if not img.endswith('.tif') else img for img in filtered_images]
            images = [img for img in filtered_images if img in self.img_labels_images]
            self.prepared_data = np.zeros((len(images), 4, 256, 256))
            print(f"Loaded {len(images)} images")
        else:
            images = self.test_images
            self.prepared_data = np.zeros((len(images), 12, 1024, 1024))

        for idx, img in enumerate(natsorted(images)):
            print('Processing', img)
            if self.train_set:
                input_path = f"{self.TRAIN_IMAGES_PATH}/{img}"
            elif self.img_labels:
                input_path = f"{self.img_labels_images_path}/{img}"
            else:
                input_path = f"{self.TEST_IMAGES_PATH}/{img}"
            data, meta, stats = self.analyze_nans(input_path)
            filled_data = self.fill_nans(data, method=method)
            self.prepared_data[idx] = filled_data

        if self.satlas:
            bands = [1, 2, 3, 4, 5, 6, 7, 10, 11]
            self.prepared_data = self.prepared_data[:, bands] #satlas model requires bands 1,2,3,4,5,6,7,10,11 or RGB
            self.prepared_data = self.normalize_sentinel2(
                    self.prepared_data[:, :3, :, :],
                    self.prepared_data[:, 3:, :, :]
                )
        else:
            self.prepared_data = self.normalize_image(self.prepared_data)
        if self.train_set:
            self.label_pixels()
            print("NaN values filled and pixels labeled")
            print("self.labels.shape:", self.labels.shape)
            

        
        print("self.prepared_data.shape:", self.prepared_data.shape)
        self.verify_no_nans()

    def save_preprocessed(self, filename='/prepared_data.npy', labels = True):
        """Save preprocessed data and labels to disk"""
        data_path = self.TRAIN_IMAGES_PATH + filename
        labels_path = self.TRAIN_IMAGES_PATH + '/labels.npy'
        np.save(data_path, self.prepared_data)
        print(f"Preprocessed data saved to {data_path}")
        if labels:
            np.save(labels_path, self.labels)
            print(f"Labels saved to {labels_path}")

    def load_preprocessed_data(self):
        """Load preprocessed images and labels from disk"""
        self.prepared_data = np.load(self.TRAIN_IMAGES_PATH + '/prepared_data.npy')
        self.labels = np.load(self.TRAIN_IMAGES_PATH + '/labels.npy')
        print(f'Loaded preprocessed data from {self.base_dir}')
        return self.prepared_data, self.labels
        


def plot_histograms(data, title_prefix, bins=50):
    """
    Plot histograms for each channel in the image dataset.

    Parameters:
      data: NumPy array of shape (N, C, H, W)
      title_prefix: Title prefix used for labeling the plots.
      bins: Number of bins in the histogram.
    """
    num_channels = data.shape[1]
    fig, axs = plt.subplots(1, num_channels, figsize=(5*num_channels, 4))

    # Make sure axs is an array even if there's one channel
    if num_channels == 1:
        axs = [axs]

    for i in range(num_channels):
        # Flatten the array for the given channel across all images
        channel_data = data[:, i, ...].flatten()
        axs[i].hist(channel_data, bins=bins, alpha=0.75, color='blue')
        axs[i].set_title(f'{title_prefix} - Channel {i+1}')
        axs[i].set_xlabel('Pixel Value')
        axs[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()



def histogram_match_images(source, reference):
    """
    Perform histogram matching for each channel (band) in 'source'
    so that it matches the pixel distribution of 'reference'.

    Parameters:
      source: np.array of shape (N, C, H, W)
      reference: np.array of shape (M, C, H, W)
    Returns:
      matched: np.array of the same shape as 'source' with adjusted pixel values
    """
    # We'll flatten all images across the batch dimension, then match histograms channel by channel.
    C = source.shape[1]

    matched = np.zeros_like(source)

    for c in range(C):
        # Flatten across batch dimension for this channel
        src_c = source[:, c, :, :].reshape(-1, source.shape[2], source.shape[3])
        ref_c = reference[:, c, :, :].reshape(-1, reference.shape[2], reference.shape[3])

        src_c_flat = src_c.reshape(-1)
        ref_c_flat = ref_c.reshape(-1)

        src_c_dummy = src_c_flat.reshape(-1, 1)
        ref_c_dummy = ref_c_flat.reshape(-1, 1)

        # Perform histogram matching
        matched_c_dummy = match_histograms(src_c_dummy, ref_c_dummy)

        # Reshape back
        matched_c = matched_c_dummy.reshape(src_c.shape)

        # Put it back into the matched array
        matched[:, c, :, :] = matched_c

    return matched

def create_bright_reference(shape=(10, 3, 64, 64), mean=0.2, std=0.05):
    """
    Create a synthetic bright reference distribution, which we will use as reference for histogram matching.

    Parameters:
      shape: Shape of the reference array (N, C, H, W)
      mean: Mean of the normal distribution (higher = brighter)
      std: Standard deviation of the normal distribution

    Returns:
      reference: NumPy array with the desired bright distribution
    """
    # Create a random bright distribution truncated between 0 and 1
    reference = np.random.normal(loc=mean, scale=std, size=shape)
    return np.clip(reference, 0, 1)  # Ensure values stay within [0, 1]

def visualize_images(image_data, num_images=5, cmap='viridis'):
  """Visualizes a subset of images from the provided NumPy array.

  Args:
    image_data: NumPy array of shape (N, C, H, W) containing image data.
    num_images: Number of images to visualize.
    cmap: Colormap to use for visualization (default is 'viridis').
  """
  num_rows = (num_images + 4) // 5  # Calculate number of rows for grid
  fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3 * num_rows))

  for i in range(num_images):
    row = i // 5
    col = i % 5
    ax = axes[row, col]

    # Reshape and transpose the image for display
    image = np.transpose(image_data[i], (1, 2, 0))

    ax.imshow(image, cmap=cmap)  # Use the specified colormap if provided
    ax.axis('off')  # Turn off axis labels

  # Remove any empty subplots in the grid
  for i in range(num_images, num_rows * 5):
    row = i // 5
    col = i % 5
    fig.delaxes(axes[row, col])

  plt.tight_layout()
  plt.show()

class SatelliteSegmentationDataset(Dataset):
    """
    Dataset class for satellite image segmentation with patch extraction
    and augmentation strategies.
    """

    def __init__(
        self,
        images: np.ndarray, 
        labels: np.ndarray, 
        patch_size: int = 256,
        patch_stride: Optional[int] = 128,
        min_valid_pixels: float = 0.05,
        augment: bool = True,
        max_patches_per_image: Optional[int] = None,
    ):
        self.images = images
        self.labels = labels
        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size
        self.min_valid_pixels = min_valid_pixels
        self.augment = augment
        self.max_patches_per_image = max_patches_per_image

        if self.augment:
          self.transform = A.Compose([
                  A.OneOf([
                      A.RandomRotate90(p=0.5),
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                  ], p=1),  # Apply one of the flips/rotations
                  A.OneOf([
                      A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
                      A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
                      A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                  ], p=1),  # Apply one of the brightness/jitter/gamma augmentations
        ])
        self.patches = self._create_patches()


    def _create_patches(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create patches from larger images.
        Returns list of (image_patch, label_patch) tuples.
        """
        patches = []
        N, C, H, W = self.images.shape

        for img_idx in range(N):
            image = self.images[img_idx]  # (C, H, W)
            label = self.labels[img_idx]  # (H, W)

            # Track valid patches for this image
            image_patches = []

            for y in range(0, H - self.patch_size + 1, self.patch_stride):
                for x in range(0, W - self.patch_size + 1, self.patch_stride):
                    img_patch = image[
                        :, y : y + self.patch_size, x : x + self.patch_size
                    ]
                    label_patch = label[
                        y : y + self.patch_size, x : x + self.patch_size
                    ]

                    # Check if patch contains enough non-background pixels
                    if self._is_valid_patch(label_patch):
                        image_patches.append((img_patch, label_patch))


            patches.extend(image_patches)

        return patches

    def _is_valid_patch(self, label_patch: np.ndarray) -> bool:
        """
        Check if patch contains enough non-background pixels.
        """
        non_background = (label_patch > 0).sum()
        total_pixels = label_patch.size
        return (non_background / total_pixels) >= self.min_valid_pixels

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_patch, label_patch = self.patches[idx]

        if self.augment:
            # Transpose image to (H, W, C) for albumentation (needed for ColorJitter)
            img_patch = np.transpose(img_patch, (1, 2, 0))
            #pixel values for augmentation
            img_patch = (img_patch * 255).astype(np.uint8)
            transformed = self.transform(image=img_patch, mask=label_patch)
            img_patch = np.transpose(transformed["image"], (2, 0, 1))
            img_patch = img_patch.astype(np.float32) / 255.0 # Convert back to float32 and normalized
            label_patch = transformed["mask"]

        return (
            torch.as_tensor(img_patch.copy(), dtype=torch.float32),
            torch.as_tensor(label_patch, dtype=torch.long),
        )

class TestDataset(Dataset):
    """Dataset class for test data with test-time augmentation"""
    def __init__(
        self,
        images: np.ndarray,  # Shape: (N, C, H, W)
        patch_size: int = 256,
        stride: int = 256,
        tta: bool = True
    ):
        self.images = images
        self.patch_size = patch_size
        self.stride = stride
        self.tta = tta
        self.patches, self.positions = self._create_patches()

        # Define test-time augmentations
        if self.tta:
            self.transforms = [
                None,  # Original image (no transform)
                A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
                A.Transpose(p=1.0),
                A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                            A.RandomGamma(gamma_limit=(80, 120), p=1.0)])

            ]
        else:
            self.transforms = [None]  # Only original image

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
        return len(self.patches) * len(self.transforms)

    def __getitem__(self, idx):
        # Calculate which patch and which augmentation
        patch_idx = idx // len(self.transforms)
        aug_idx = idx % len(self.transforms)

        # Get the original patch
        patch = self.patches[patch_idx]

        # Apply transformation if not None
        if self.transforms[aug_idx] is not None:
            # Convert from pytorch format (C,H,W) to numpy (H,W,C)
            patch_np = np.transpose(patch, (1, 2, 0))
            transformed = self.transforms[aug_idx](image=patch_np)
            transformed_patch = np.transpose(transformed['image'], (2, 0, 1))
            return torch.as_tensor(transformed_patch, dtype=torch.float32)
        else:
            return torch.as_tensor(patch, dtype=torch.float32)

def create_dataloaders(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int = 8,
    patch_size: int = 256,
    patch_stride: int = 128,
    num_workers: int = 0,
    num_classes = 5,

) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders with the custom dataset class.
    """
    train_dataset = SatelliteSegmentationDataset(
        images=train_images,
        labels=train_labels,
        patch_size=patch_size,
        patch_stride=patch_stride,
        augment=True,
    )

    val_dataset = SatelliteSegmentationDataset(
        images=val_images,
        labels=val_labels,
        patch_size=patch_size,
        patch_stride=patch_stride,
        augment=False
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers=num_workers,
        shuffle = True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader

class SimpleHead(nn.Module):
    def __init__(self, fpn_channels, num_categories=5):
        """
        Args:
            fpn_channels: List of tuples which denotes spatial dimension and feature channels for each FPN output
            num_categories: Total number of output channels (background + 4 foreground classes).
        """
        super(SimpleHead, self).__init__()
        self.num_outputs = num_categories
        #this is the same implementation as in the original satlaspretrain code
        use_channels = fpn_channels[0][1]
        num_layers = 2
        layers = []
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(use_channels, use_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ]
        self.layers = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(use_channels, self.num_outputs, kernel_size=3, padding=1)


    def forward(self, image_list, raw_features, targets=None):
        """
        Args:
            image_list: (Not used here.)
            raw_features: A list of 5 feature tensors from final upsampling
        Returns:
            raw_outputs: Segmentation logits from decoder, shape [B, 5, H, W].
        """
        x_dec = self.layers(raw_features[0])
        raw_outputs = self.final_conv(x_dec) 

        return raw_outputs, None

def compute_f1_score(pred_mask: np.ndarray, truth_mask: np.ndarray) -> float:
    """
    Compute pixel-based F1 for two binary masks.
    """
    tp = ((pred_mask > 0) & (truth_mask > 0)).sum()
    fp = ((pred_mask > 0) & (truth_mask == 0)).sum()
    fn = ((pred_mask == 0) & (truth_mask > 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

def performance_df_bhw(
    pred_labels: np.ndarray,
    gt_labels:   np.ndarray,
    class_names: List[str] = ["background", "plantation", "grassland", "mining", "logging"],
    min_area:     int = 1000
) -> pd.DataFrame:
    """
    Args:
        pred_labels: array [B, H, W] of int labels 0..C-1
        gt_labels:   array [B, H, W] of int labels 0..C-1
        class_names: list of class names, length = C
        min_area:    minimum pixel area to keep a class mask
    Returns:
        DataFrame: per-image (rows), per-class (cols) F1, plus 'all_classes' col and 'all_images' row.
        calling performance_df_bhw(pred_labels, gt_labels).loc['all_images','all_classes'] 
        will return average F1-score for entire batch.
    """
    B, H, W = pred_labels.shape
    C = len(class_names)

    rows = []
    for i in range(B):
        row = {}
        valid_classes = []
        for c, name in enumerate(class_names):
            if c == 0:  # Skip background class (0)
                continue
            pm = (pred_labels[i] == c)
            tm = (gt_labels[i] == c)
            if pm.sum() < min_area:
                pm = np.zeros((H, W), dtype=bool)
            if pm.sum() == 0 and tm.sum() == 0:
                continue  # Skip classes absent in both prediction and ground truth
            row[name] = compute_f1_score(pm, tm)
            valid_classes.append(name)
        if valid_classes:
            row["all_classes"] = np.mean([row[name] for name in valid_classes])
        else:
            row["all_classes"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows, index=[f"img_{i}" for i in range(B)])
    df.loc["all_images"] = df.mean(numeric_only=True)
    return df


class CustomLoss(nn.Module):
    def __init__(self, delta=1.0, num_classes=5):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.robust_loss = RobustLoss(delta=delta, num_classes=num_classes)
        self.binary_focal = smp.losses.FocalLoss(mode='binary')

    def forward(self, logits, targets):
        bg_target = (targets == 0).float()  # Background/foreground mask

        # Background logits
        bg_logits_dec = logits[:, 0, :, :]  # Logits for the background class

        # Calculate binary focal loss for background/foreground
        decoder_loss_bg = self.binary_focal(bg_logits_dec.unsqueeze(1), bg_target.unsqueeze(1))
        # Calculate robust loss for all classes, but background is not perturbed
        decoder_loss_roi = self.robust_loss(logits, targets)

        decoder_loss = (2.0 * decoder_loss_bg) + decoder_loss_roi

        return decoder_loss
    
class RobustLoss(nn.Module):

  def __init__(self, delta=1.0, num_classes=5):
      """
      Initializes the robust loss with a penalty factor delta.

      Args:
          delta (float): The penalty factor. A positive value which reduces the
                          logit of the correct class and boosts those of the incorrect classes.
          num_classes (int): The number of classes in the segmentation task.
      """
      super(RobustLoss, self).__init__()
      self.delta = delta
      self.num_classes = num_classes

  def forward(self, logits, targets):
      # Convert targets to one-hot encoding
      targets_onehot = F.one_hot(targets, num_classes=self.num_classes) 
      targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

      # Create a mask: 0 for background channel, 1 for foreground channels in order to not perturb background
      mask = torch.ones_like(targets_onehot)
      mask[:, 0, :, :] = 0 

      perturbed_logits = logits + self.delta * mask * (1 - 2 * targets_onehot)

      perturbed_logits = perturbed_logits.contiguous()
      targets_onehot = targets_onehot.contiguous()

      loss = F.cross_entropy(perturbed_logits, targets)

      return loss

def plot_loss(train_losses, val_losses):
    """Plots training and validation losses.

    Args:
        train_losses: A list of training losses for each epoch.
        val_losses: A list of validation losses for each epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



def predict_probs(model, test_dataloader, device='cpu'):
    model.eval()
    tta = hasattr(test_dataloader.dataset, 'tta') and test_dataloader.dataset.tta
    num_transforms = len(test_dataloader.dataset.transforms) if tta else 1

    # Get total number of patches
    num_patches = len(test_dataloader.dataset.patches)

    # Initialize output numpy array with correct shape
    patch_size = test_dataloader.dataset.patch_size
    num_classes = 5 
    all_probs = np.zeros((num_patches, patch_size, patch_size, num_classes), dtype=np.float32)

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_dataloader, desc="Running inference")):
            data = data.to(device)
            output = model(data)[0]
            probs = torch.nn.functional.softmax(output, dim=1)
            batch_probs = probs.permute(0, 2, 3, 1).cpu().numpy()

            # Process each sample in the batch
            for i, pred in enumerate(batch_probs):
                # Determine patch index
                global_idx = batch_idx * test_dataloader.batch_size + i
                patch_idx = global_idx // num_transforms if tta else global_idx

                # Apply inverse transformation if using TTA
                if tta:
                    transform_idx = global_idx % num_transforms
                    if transform_idx == 0:  # Original
                        aligned_pred = pred
                    elif transform_idx == 1:  # HorizontalFlip + VerticalFlip
                        aligned_pred = pred[::-1, ::-1, :]
                    elif transform_idx == 2:  # Transpose
                        aligned_pred = np.transpose(pred, (1, 0, 2))
                    else:
                        aligned_pred = pred  # Default case

                    all_probs[patch_idx] += aligned_pred

                    # Average after each transform
                    if transform_idx == num_transforms - 1:
                        all_probs[patch_idx] /= num_transforms
                else:
                    all_probs[patch_idx] = pred  # Directly assign if no TTA

    return all_probs


def stitch_patches(prob_patches, positions, image_shape=(1024, 1024), patch_size=256):
    """
    Stitch patches back together while handling overlapping predictions from striding

    Args:
        prob_patches:
         Tensor of shape (N, patch_size, patch_size, num_classes)
        positions: List of tuples (image_idx, y, x) for each patch
        image_shape: Tuple of (height, width) for final image
        patch_size: Size of each patch
    """
    num_classes = prob_patches.shape[-1]
    image_indices = sorted(set(pos[0] for pos in positions))
    stitched_images = []

    for img_idx in image_indices:
        # Initialize accumulator tensors
        prob_sum = np.zeros((image_shape[0], image_shape[1], num_classes))
        count = np.zeros((image_shape[0], image_shape[1], 1))

        # Place all patches for current image
        for i, (curr_idx, y, x) in enumerate(positions):
            if curr_idx == img_idx:
                # Add probabilities
                prob_sum[y:y+patch_size, x:x+patch_size, :] += prob_patches[i]
                # Increment counter for averaging
                count[y:y+patch_size, x:x+patch_size, :] += 1

        # Average overlapping regions
        count = np.maximum(count, 1)  # Avoid division by zero
        full_image = prob_sum / count

        stitched_images.append(torch.from_numpy(full_image))

    return stitched_images



def post_process_torch(prob_maps,
                      kernel_size=5,
                      min_size=100,
                      border_margin=10,
                      apply_smoothing=True,
                      _remove_small_objects=True,
                      fill_enclosed=False,
                      smooth_boundaries=True,
                      ):
    """
    Post-process segmentation maps with optional processing steps.

    Args:
        prob_maps: List of tensors, each [1024, 1024, 5] with class probabilities
        kernel_size: Size of Gaussian kernel for smoothing
        min_size: Minimum connected component size
        border_margin: Pixels from edge to ignore enclosed region rule
        apply_smoothing: Apply Gaussian smoothing
        remove_small_objects: Remove small connected components
        fill_enclosed: Fill enclosed background regions
        smooth_boundaries: Apply boundary smoothing
    Returns:
        List of processed segmentation maps, each [1024, 1024]
    """
    processed_maps = []

    # Create 2D Gaussian kernel if needed
    if apply_smoothing:
        sigma = kernel_size / 6.0
        center = kernel_size // 2
        x, y = np.mgrid[0:kernel_size, 0:kernel_size]
        kernel_2d = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = torch.from_numpy(kernel_2d).double().view(1, 1, kernel_size, kernel_size)

    for prob_map in prob_maps:
        if not isinstance(prob_map, torch.Tensor):
            prob_map = torch.from_numpy(prob_map)
        prob_map = prob_map.double()

        # Step 1: Optional smoothing
        if apply_smoothing:
            smoothed_probs = []
            for c in range(prob_map.shape[-1]):
                class_prob = prob_map[..., c].unsqueeze(0).unsqueeze(0)
                smoothed = F.conv2d(class_prob, kernel_2d, padding=kernel_size//2)
                smoothed_probs.append(smoothed.squeeze())
            smoothed_probs = torch.stack(smoothed_probs, dim=-1)
        else:
            smoothed_probs = prob_map

        pred_np = torch.argmax(smoothed_probs, dim=-1).cpu().numpy()

        # Step 2: Optional enclosed region handling
        if fill_enclosed:
            h, w = pred_np.shape
            inner_mask = np.zeros((h, w), dtype=bool)
            inner_mask[border_margin:-border_margin, border_margin:-border_margin] = True

            background = (pred_np == 0)
            bg_labels, num_labels = ndimage.label(background)

            for label in range(1, num_labels + 1):
                component = (bg_labels == label)
                if not np.any(component & ~inner_mask):
                    dilated = ndimage.binary_dilation(component)
                    border = dilated & ~component
                    surrounding_classes = pred_np[border]
                    if len(np.unique(surrounding_classes)) == 1 and surrounding_classes[0] != 0:
                        pred_np[component] = surrounding_classes[0]

        # Step 4: Optional small object removal
        if _remove_small_objects:
            for class_idx in range(1, smoothed_probs.shape[-1]):
                class_mask = (pred_np == class_idx)
                if class_mask.any():
                    filtered_mask = remove_small_objects(class_mask, min_size=min_size)
                    pred_np[class_mask & ~filtered_mask] = 0

        # Step 5: Optional boundary smoothing
        if smooth_boundaries:
            for class_idx in range(1, smoothed_probs.shape[-1]):
                class_mask = (pred_np == class_idx)
                if class_mask.any():
                    closed_mask = ndimage.binary_closing(class_mask, structure=np.ones((3,3)))
                    pred_np[class_mask | closed_mask] = (
                        class_idx * (closed_mask[class_mask | closed_mask]))


        processed_maps.append(torch.from_numpy(pred_np))

    return processed_maps



def converter(tensors):
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


def write_json(polygons, filename: str):
    """
    Function that takes a list of polygons and writes them to a JSON file.
    The list of polygons is a list of lists of polygons, where each polygon is a list of points.
    Dictionary to map unique values to class names.
    """
    numper_2_class = {1: "plantation", 2: "grassland_shrubland", 3: "mining", 4: "logging"}

    # Construct the full file path within Google Drive
    save_path = os.path.join('/content/drive/MyDrive/inf367/', filename + '.json')  # Modify if you want to save it somewhere else in Drive

    # Create necessary directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as file:
        images_overview = {"images": []}
        # Iterate over images in polygons
        for i in range(len(polygons)):
            curr = {"file_name": f"evaluation_{i}.tif", "annotations": []}
            # Iterate over types of polygons in image
            for j in range(1, len(polygons[i])):
                # Convert polygons to list of coordinates
                for k in range(len(polygons[i][j])):
                    listed_polygons = []
                    for p in range(len(polygons[i][j][k])):
                        listed_polygons.append(polygons[i][j][k][p][0])
                        listed_polygons.append(polygons[i][j][k][p][1])
                    curr["annotations"].append({"class": numper_2_class[j], "segmentation": listed_polygons})
            images_overview["images"].append(curr)
        file.write(json.dumps(images_overview, ensure_ascii=False, indent=4))


def display_labels(post_processed, image=0):
    """
    Function to display the predicted labels for an image.

    Args:
        post_processed: Predictions array/tensor
        image: Index of image to display (default=0)
    """
    # Ensure we're working with numpy array
    if isinstance(post_processed[image], torch.Tensor):
        prediction = post_processed[image].cpu().detach().numpy()
    else:
        prediction = post_processed[image]

    # Color mapping for classes
    color_map = {
        0: [0, 0, 0],      # background: black
        1: [255, 0, 0],    # plantation: red
        2: [0, 255, 0],    # grassland_shrubland: green
        3: [0, 0, 255],    # mining: blue
        4: [255, 0, 255]   # logging: purple
    }

    # Create RGB image
    height, width = prediction.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Vectorized operation for coloring
    for class_id, color in color_map.items():
        mask = (prediction == class_id)
        rgb_image[mask] = color

    # Display with matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.axis('off')

    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=np.array(color)/255)
                      for color in color_map.values()]
    legend_labels = ['Background', 'Plantation', 'Grassland/Shrubland', 'Mining', 'Logging']
    plt.legend(legend_elements, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()