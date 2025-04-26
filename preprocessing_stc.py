#imports
import os
import json
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import tifffile as tiff
import cv2
from PIL import Image, ImageDraw
from natsort import natsorted

class PreProcessing:
    # structure: {class name: (color, RGB, class label)
    annotation_color_allocation = {
    'background' : ('black', (0, 0, 0), 0), 
    'plantation' : ('red', (255, 0, 0), 1), 
    'grassland_shrubland' : ('green', (0, 255, 0), 2), 
    'mining' : ('blue', (0, 0, 255), 3), 
    'logging' : ('yellow', (255, 255, 0), 4) 
}   
    def __init__(self, train_set=True):
        #laste inn annotations fra JSON og bilder fra fil
        self.base_dir = os.getcwd()
        self.parent_dir = "/content/drive/MyDrive/Colab_Notebooks/INF367A/"
        self.TRAIN_IMAGES_PATH=self.parent_dir+"train_images"
        self.TEST_IMAGES_PATH=self.parent_dir+"evaluation_images" 
        # plot and save RGB with annotation
        with open('/content/drive/MyDrive/Colab_Notebooks/INF367A/train_annotations.json', 'r') as file:
            self.train_annotations = json.load(file)

        self.train_set=train_set
        self.polygons = {}
        print(self.TRAIN_IMAGES_PATH)
        self.train_images = [f for f in os.listdir(self.TRAIN_IMAGES_PATH) if f.endswith('.tif')]
        self.test_images = [f for f in os.listdir(self.TEST_IMAGES_PATH) if f.endswith('.tif')]
        for image in self.train_annotations["images"]:
            polys=[]
            for polygons in image["annotations"]:
                geom = np.array(polygons['segmentation'])
                polys.append((polygons["class"],geom))
            self.polygons[image["file_name"]]=polys

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
    
    def visualize_in_RGB(self, n=10, start=0):
        #Velg train_image_num for alle bilder
        #n=train_image_num
        outfolder = 'rgb_annotation'  
        for index_for_image in range(start, n):
            filename = os.path.join(outfolder, f'rgb_with_annotation_{index_for_image}.png')
            if os.path.exists(filename):
                print(f"File {filename} already exists. Skipping.")
                continue
            # Visualize sample tif data
            SAMPLE_TIF_PATH = f'{self.TRAIN_IMAGES_PATH}/train_{index_for_image}.tif' if self.train_set else f'{self.TEST_IMAGES_PATH}/evaluation_{index_for_image}.tif'
            annotation_data = self.train_annotations['images']
            id_to_annotation = {item['file_name']: item for item in annotation_data}
            annotation_data = id_to_annotation[ f'train_{index_for_image}.tif']['annotations']

            #annotation_data = train_annotations['images'][index_for_train_image]['annotations']
            
            # Convert to GeoJSON
            geojson_data = self.convert_to_geojson(annotation_data)
            gdf = gpd.GeoDataFrame.from_features(geojson_data)
            # Open sample tif file
            with rasterio.open(SAMPLE_TIF_PATH) as src:
                # Read bands 2, 3, and 4 (B, G, R)
                b2 = src.read(2)
                b3 = src.read(3)
                b4 = src.read(4)
                #Checking that all bands have the same shape (1024,1024)
                #for band in range(1,13):
                #    print(band, src.read(band).shape)
                # Stack bands to create a 3D array (height, width, channels)
                rgb_image = np.dstack((b4, b3, b2))

                rgb_image = np.nan_to_num(rgb_image) # Replace NaN and inf with 0

                # Normalize pixel values for display
                rgb_image = rgb_image.astype(np.float32)
                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
                rgb_image = np.clip(rgb_image, 0, 1) # Clip values to 0-1 range for floats
                """
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(rgb_image)
                
                for idx in range(len(gdf)):
                    # print(gdf.iloc[idx]['class'])
                    gdf.iloc[[idx]].boundary.plot(ax=ax, color=PreProcessing.annotation_color_allocation[gdf.iloc[idx]['class']][0])

                # Create and add custom legend
                legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                                            markerfacecolor=value[0], markersize=10) 
                                    for key, value in PreProcessing.annotation_color_allocation.items() 
                                    if key in gdf['class'].unique()]
                #ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

                # gdf.boundary.plot(ax=ax, color='red')
                #plt.title(f'idx {index_for_image}: RGB Image with annotation')
                """
                #Om man vil lagre
                #plt.savefig(filename)
                #plt.show()


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
                filled = self.fill_nans(original[np.newaxis, ...], method=method)[0]  # Add band dim
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                im1 = ax1.imshow(original, cmap='viridis')
                ax1.set_title('Original with NaNs')
                plt.colorbar(im1, ax=ax1)
                
                im2 = ax2.imshow(filled, cmap='viridis')
                ax2.set_title(f'After {method} Filling')
                plt.colorbar(im2, ax=ax2)
                
                plt.suptitle(f'train_{idx}.tif - Band {band}')
                plt.tight_layout()
                """ output_png = tif_path.replace('.tif', '_comparison.png')
                plt.savefig(output_png)
                plt.show() """
    
    def create_polygons(self, json_file='train_annotations.json'):
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
        json_file = os.path.join(
                '/content/drive/MyDrive/Colab_Notebooks/INF367A/',
                'train_annotations.json'
            )
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
    
    
    def normalize_sentinel2(self, tci_rgb, bands, batch_size=64):
        """Normalize Sentinel-2 imagery with memory-efficient batch processing"""
        # Get input shapes
        n_samples = tci_rgb.shape[0]
        height = tci_rgb.shape[2]
        width = tci_rgb.shape[3]
        
        # Pre-allocate output array preserving spatial dimensions
        n_channels = tci_rgb.shape[1] + bands.shape[1]
        output = np.empty((n_samples, n_channels, height, width), dtype=np.float32)
        
        # Process in batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Process TCI batch preserving spatial dimensions
            tci_batch = tci_rgb[start_idx:end_idx]
            tci_min = tci_batch.min(axis=(2,3), keepdims=True)
            tci_range = tci_batch.max(axis=(2,3), keepdims=True) - tci_min
            tci_batch = (tci_batch - tci_min) / tci_range
            
            # Process bands batch preserving spatial dimensions
            bands_batch = bands[start_idx:end_idx] / 8160.0
            np.clip(bands_batch, 0, 1, out=bands_batch)
            
            # Store in output array
            output[start_idx:end_idx, :tci_rgb.shape[1]] = tci_batch
            output[start_idx:end_idx, tci_rgb.shape[1]:] = bands_batch
            
            # Free memory
            del tci_batch
            del bands_batch
            
        return output      

    def preprocess(self, method='open_cv_inpaint_telea'):
        selected_bands = [2, 3, 4]  # B, G, R (1-based indexing for rasterio)
        images = self.train_images if self.train_set else self.test_images
        self.prepared_data = np.zeros((len(images), len(selected_bands), 1024, 1024), dtype=np.float32)
        print(natsorted(images))
        for idx, img in enumerate(natsorted(images)):
            print('Processing', img)
            input_path = f'{self.TRAIN_IMAGES_PATH}/{img}' if self.train_set else f'{self.TEST_IMAGES_PATH}/{img}'

            with rasterio.open(input_path) as src:
                bands_data = []
                for band in selected_bands:
                    band_data = src.read(band)
                    bands_data.append(band_data)
                data = np.stack(bands_data)

            filled_data = self.fill_nans(data, method=method)
            self.prepared_data[idx] = filled_data

        # Normalize only the 3 bands we loaded (no extra bands)
        self.prepared_data = self.normalize_sentinel2(self.prepared_data, np.empty((len(images), 0, 1024, 1024)))

        if self.train_set:
            self.label_pixels()
            print("NaN values filled and pixels labeled")
            print("self.labels.shape:", self.labels.shape)

        print("self.prepared_data.shape:", self.prepared_data.shape)
      
    def save_preprocessed(self):
        """Save preprocessed data and labels to disk"""

        data_path = self.TEST_IMAGES_PATH + '/prepared_eval_data.npy'
        #labels_path = self.TEST_IMAGES_PATH + '/labels.npy'
        np.save(data_path, self.prepared_data)
        print(f"Preprocessed data saved to {data_path}")
        #print(f"Labels saved to {labels_path}")


    def load_preprocessed_data(self):
        """Load preprocessed images and labels from disk"""
        self.prepared_data = np.load(self.TRAIN_IMAGES_PATH + '/prepared_data.npy')
        self.labels = np.load(self.TRAIN_IMAGES_PATH + '/labels.npy')
        print(f'Loaded preprocessed data from {self.base_dir}')
        return self.prepared_data, self.labels

