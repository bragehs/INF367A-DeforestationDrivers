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

class ProcessData:
    # structure: {class name: (color, RGB, class label)
    annotation_color_allocation = {
    'background' : ('black', (0, 0, 0), 0), 
    'plantation' : ('red', (255, 0, 0), 1), 
    'grassland_shrubland' : ('green', (0, 255, 0), 2), 
    'mining' : ('blue', (0, 0, 255), 3), 
    'logging' : ('yellow', (255, 255, 0), 4) 
}   
    def __init__(self):
        #laste inn annotations fra JSON og bilder fra fil
        self.base_dir = os.getcwd()
        self.parent_dir = os.path.split(self.base_dir)[0]
        self.TRAIN_IMAGES_PATH=self.parent_dir+"/train_images" 

        # plot and save RGB with annotation
        with open('train_annotations.json', 'r') as file:
            self.train_annotations = json.load(file)

        self.polygons = {}
        self.images = []
        for image in self.train_annotations["images"]:
            polys=[]
            for polygons in image["annotations"]:
                geom = np.array(polygons['segmentation'])
                polys.append((polygons["class"],geom))
            self.polygons[image["file_name"]]=polys

            self.images.append(image["file_name"])

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
        for index_for_train_image in range(start, n):
            filename = os.path.join(outfolder, f'rgb_with_annotation_{index_for_train_image}.png')
            if os.path.exists(filename):
                print(f"File {filename} already exists. Skipping.")
                continue
            # Visualize sample tif data
            SAMPLE_TIF_PATH = f'{self.TRAIN_IMAGES_PATH}/train_{index_for_train_image}.tif'
            annotation_data = self.train_annotations['images']
            id_to_annotation = {item['file_name']: item for item in annotation_data}
            annotation_data = id_to_annotation[ f'train_{index_for_train_image}.tif']['annotations']

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

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(rgb_image)
                ax.invert_yaxis()
                
                for idx in range(len(gdf)):
                    # print(gdf.iloc[idx]['class'])
                    gdf.iloc[[idx]].boundary.plot(ax=ax, color=ProcessData.annotation_color_allocation[gdf.iloc[idx]['class']][0])

                # Create and add custom legend
                legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                                            markerfacecolor=value[0], markersize=10) 
                                    for key, value in ProcessData.annotation_color_allocation.items() 
                                    if key in gdf['class'].unique()]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

                # gdf.boundary.plot(ax=ax, color='red')
                plt.title(f'idx {index_for_train_image}: RGB Image with annotation')

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
            mask = np.isnan(band).astype(np.uint8)
            
            if method == 'open_cv_inpaint_telea' or method == 'open_cv_inpaint_ns':
                # Replace NaNs with 0 for inpainting
                band[mask == 1] = 0
                filled = cv2.inpaint(
                    band.astype(np.float32),
                    mask * 255,
                    inpaintRadius=kwargs.get('inpaint_radius', 3),
                    flags=cv2.INPAINT_TELEA if method == 'opencv_inpaint_telea' else cv2.INPAINT_NS
                )
            
            filled_data[b] = filled
        
        return filled_data

    def visualize_nan_filling(self, start=0, n=1, band_start=1, band_n=12, method= "open_cv_inpaint_telea"):
        """Visualize NaN filling results for a specific band."""
        for idx in range(start, n):
            for band in range(band_start, band_n +1):
                SAMPLE_TIF_PATH = f'{self.TRAIN_IMAGES_PATH}/train_{idx}.tif'
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
    
    def rasterize(self):
                #Test mask
        shape = (1024, 1024)
        self.RGB_raster_imgs={}
        for current_image in range(len(self.train_annotations['images'])):
            Image.MAX_IMAGE_PIXELS = None
            img = Image.new('RGB', (shape[1], shape[0]), (0, 0, 0))  # (w, h)

            for i in range(len(self.polygons[f"train_{current_image}.tif"])):

                poly = self.polygons[f"train_{current_image}.tif"][i][1]
                type_deforest= self.polygons[f"train_{current_image}.tif"][i][0]

                points = list(zip(poly[::2], poly[1::2]))
                points = [(x, y) for x, y in points]
                color = ProcessData.annotation_color_allocation[type_deforest][1]
                ImageDraw.Draw(img).polygon(points, outline=None, fill=color)
            mask_2 = np.array(img)
            self.RGB_raster_imgs[f"train_{current_image}.tif"]=mask_2
    
    def visualize_rasterized(self, start=0, n=10):
        self.rasterize()
        for i in range(start, n):
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            plt.imshow(self.RGB_raster_imgs[f"train_{i}.tif"])

        # Add padding around the actual image
            plt.subplots_adjust(right=0.85)
            
            # Create legend text in whitespace
            legend_x = 1.02  # Position outside of image

            # Add legend items with consistent spacing
            for idx, (label, color) in enumerate(ProcessData.annotation_color_allocation.items()):
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
        for k, v in ProcessData.annotation_color_allocation.items()
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

    def preprocess(self, method='open_cv_inpaint_telea'):
        """Process TIF file and save result"""
        self.prepared_data = np.zeros((len(self.images), 12, 1024, 1024))
        for idx, img in enumerate(natsorted(self.images)):
            print('Processing', img)
            input_path = f'{self.TRAIN_IMAGES_PATH}/{img}'
            data, meta, stats = self.analyze_nans(input_path)
            if stats['nan_count'] > 0:
                self.prepared_data[idx] = self.fill_nans(data, method=method)



        self.label_pixels()

        print("NaN values filled and pixels labeled")
        print("self.prepared_data.shape:", self.prepared_data.shape)
        print("self.labels.shape:", self.labels.shape)

    def save_preprocessed(self):
        """Save preprocessed data and labels to disk"""
        data_path = self.TRAIN_IMAGES_PATH + '/prepared_data.npy'
        labels_path = self.TRAIN_IMAGES_PATH + '/labels.npy'
        np.save(data_path, self.prepared_data)
        np.save(labels_path, self.labels)
        print(f"Preprocessed data saved to {data_path}")
        print(f"Labels saved to {labels_path}")


    def load_preprocessed_data(self):
        """Load preprocessed images and labels from disk"""
        self.prepared_data = np.load(self.TRAIN_IMAGES_PATH + '/prepared_data.npy')
        self.labels = np.load(self.TRAIN_IMAGES_PATH + '/labels.npy')
        print(f'Loaded preprocessed data from {self.base_dir}')
        return self.prepared_data, self.labels