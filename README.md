## Folder structure: 
##### inf367: 
folder with all files in this repository, and train_annotations from solafune
##### train_images: 
folder with train images from solafune, in parent directory of inf367
##### evaluation_images: 
folder with evaluation images from solafune, in parent directory of inf367
##### train-tif-v2: 
folder with train_images from Kaggle, in parent directory of inf367
##### train_v2.csv:
folder with image labels from Kaggle,  in parent directory of inf367

## Original Approach

### original_utils.py

Contains all the classes and functions used in original approach. Includes everything from preprocessing to predictions. Some of the methods are also imported for use in later implementations. This file does not need be explicitly ran. 

### prepare_dataset.ipynb

Notebook which contains data exploration and preparation of the Solafune dataset. Needs to be run from top to bottom in order to preprocess and save the train images, train labels and evaluation images into .npy files. 

### original_implementation.ipynb

From top to bottom, this notebook runs hyperparameter tuning, final training, and finally writes a json file which can be submitted to solafune competition. prepare_dataset needs to be ran before running this notebook. 

## PseudoSeg

### prepare_additional_dataset.ipynb

Notebook which prepares the Kaggle dataset that is leveraged as weakly supervised data. It maps from labels in kaggle dataset, to corresponding (or similar) labels in solafune dataset. Saves 1846 images to a .npy file, and saves an dataframe with labels in equal length.

## PseudoSeg_implementation.ipynb

 From top to bottom, this notebook also runs hyperparameter tuning, final training, and finally writes a submittable json file. Similar in many ways to original_implementation, and it uses many of the same functions. But some additional classes and functions are required in order to implement the PseudoSeg method. PseudoSegHead class contains the actual method. 


