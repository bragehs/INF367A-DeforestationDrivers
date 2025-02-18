from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
from rasterio.features import rasterize
import json

#CODE from solafune-tools
class PixelBasedMetrics:
    def __init__(self) -> None:
        pass
    def polygons_to_mask(self, polygons, array_dim):
        """
        Converts a list of polygons into a binary mask.
        
        Args:
            polygons (list): List of polygons, where each polygon is represented by a list of (x, y) tuples.
            array_dim (tuple): Dimensions of the output mask (height, width).
        
        Returns:
            np.ndarray: Binary mask with 1s for polygon areas and 0s elsewhere.
        """
        shapes = [(polygon, 1) for polygon in polygons]
        mask = rasterize(shapes, out_shape=array_dim, fill=0, dtype=np.uint8)
        return mask
    def compute_f1(self, gt_polygons, pred_polygons, array_dim=(1024, 1024)):
        """
        Compute the F1 score, precision, and recall for the given ground truth and predicted polygons.
        
        Args:
            gt_polygons (list): List of ground truth polygons.
            pred_polygons (list): List of predicted polygons.
            array_dim (tuple, optional): Dimensions of the output mask (height, width). Defaults to (1024, 1024).

        Returns:
            tuple: A tuple containing the F1 score, precision, and recall.
        """
        # Pixel-level improvement
        # Create binary masks for ground truth and predictions
        gt_mask = self.polygons_to_mask(gt_polygons, array_dim)
        pred_mask = self.polygons_to_mask(pred_polygons, array_dim)
        
        # Calculate pixel-level True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp = np.sum((gt_mask == 1) & (pred_mask == 1))
        fp = np.sum((gt_mask == 0) & (pred_mask == 1))
        fn = np.sum((gt_mask == 1) & (pred_mask == 0))
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # if no prediction, precision is considered as 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # if no ground truth, recall is considered as 1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # if either precision or recall is 0, f1 is 0
        
        return f1, precision, recall



    #Function which return the total F1 score for the training data
    def f1_scorer(self, path_ground, path_pred):
        with open(f"{path_ground}.json", 'r') as file:
            train_annotations = json.load(file)
        with open(f"{path_pred}.json", 'r') as file:
            pred_annotations = json.load(file)
        score=[]
        for i in range(len(train_annotations["images"])):
            gt_polygons = {'plantation' : [],
            'grassland_shrubland': [],
            'mining' :[],
            'logging':[]}
            pred_polygons = {'plantation' : [],
            'grassland_shrubland': [],
            'mining' :[],
            'logging':[]}
            for j in range(len(train_annotations["images"][i]["annotations"])):
                unproccesed_poly=train_annotations["images"][i]["annotations"][j]["segmentation"]
                processed_poly= Polygon(list(zip(unproccesed_poly[::2], unproccesed_poly[1::2])))
                gt_polygons[train_annotations["images"][i]["annotations"][j]["class"]].append(processed_poly)

            for j in range(len(pred_annotations["images"][i]["annotations"])):
                unproccesed_poly=pred_annotations["images"][i]["annotations"][j]["segmentation"]
                processed_poly= Polygon(list(zip(unproccesed_poly[::2], unproccesed_poly[1::2])))
                pred_polygons[pred_annotations["images"][i]["annotations"][j]["class"]].append(processed_poly)
            for key in gt_polygons.keys():

                if len(gt_polygons[key]) == 0 and len(pred_polygons[key]) == 0:
                    #score.append(1)
                    continue
                if len(gt_polygons[key]) == 0 or len(pred_polygons[key]) == 0:
                    score.append(0)
                    continue
                f1, precision, recall = self.compute_f1(gt_polygons[key], pred_polygons[key])
                score.append(f1)
            
        return sum(score)/len(score)