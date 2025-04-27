#  SolaFune contest: Identifying Deforestation Drivers

Deforestation is the process of removing large areas of forests in order to make room for some non-forest use. Excessive deforestation has negative impacts for its surroundings, and also contributes to climate change in general. It is a major environmental issue, and in places like the Amazonas, we have seen this excessive deforestation take place over many years. This issue has motivated a lot of research, perhaps especially in deep learning. Training neural networks on satellite images can make them understand where deforestation is taking place, and what is replacing it. Satellite imagery is not always easy to understand for humans either, which makes neural networks very useful.  Solafune hosted a competition where the goal was to provide a useful model, and the winners were determined by pixel-wise F1 score.   

Link to the contest can be found [here](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=about&tab=overview)

Applied machine learning paper for this project here: "Applied_ML_paper.pdf"

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

Contains all the classes and functions used in original approach. Includes everything from preprocessing to predictions. Some of the methods are also imported for use in other implementations. This file does not need be explicitly ran. 

### prepare_dataset.ipynb

Notebook which contains data exploration and preparation of the Solafune dataset. Needs to be run from top to bottom in order to preprocess and save the train images, train labels and evaluation images into .npy files. 

### original_implementation.ipynb

From top to bottom, this notebook runs hyperparameter tuning, final training, and finally writes a json file which can be submitted to solafune competition. prepare_dataset needs to be ran before running this notebook. 

## PseudoSeg

PseudoSeg is a semi-supervised method, which tries to leverage unsupervised / weakly supervised data in addition to standard supervised data. It uses a refined version of Grad-CAM, gradient class activation maps, and generates pseudo labels by combining these maps with the standard decoder output. The refinement process is trained on supervised data, and the pseudo labels are used to train with unsupervised / weakly supervised data. We chose to implement this method because of the size of our supervised dataset, and because we found a dataset from an old [Kaggle competition](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/data) which seemed promising. 

### prepare_additional_dataset.ipynb

Notebook which prepares the Kaggle dataset that is leveraged as weakly supervised data. It maps from labels in kaggle dataset, to corresponding (or similar) labels in solafune dataset. Saves 1846 images to a .npy file, and saves an dataframe with labels in equal length. All code in this notebook is only used in PseudoSeg implementation. 

### PseudoSeg_implementation.ipynb

 From top to bottom, this notebook also runs hyperparameter tuning, final training, and finally writes a submittable json file. Similar in many ways to original_implementation, and it uses many of the same functions. But some additional classes and functions are required in order to implement the PseudoSeg method. PseudoSegHead class contains the actual method. All code in this notebook is only used for PseudoSeg implementation. 

## Segment-Then-Classify

STC is a is a strategy for doing instance segmentation. It's made of two parts, one model which generates the segments and a second model which classifies the segments. This method seems promising as the competitions dataset was rather sparse. 

### STC.ipynb

This notebook contains the entire pipeline for STC. From generating training data from the training images. To training a classifier, using SAM on the evaluation images and finaly classifying on these segments. Some lines needs to be commentented in/out, since the paths from the already trained mode is local. The results from this method was deemed unsatisfactory. 

### preprocessing.py 
Contains several utils functions used in the baseline, but some have been altered to fit the STC pipeline.

## Segment Anything 2 (SAM2)

Segment Anything 2 (SAM2) is a promptable foundation model for image segmentation. This approach uses predictions from the PseudoSeg model as prompts for SAM2 to potentially refine segmentation masks.

### sam2.ipynb

Contains the pipeline for the SAM2 refinement process. It first clones the SAM2 repository. Then, it loads the evaluation images and the corresponding PseudoSeg prediction masks. These masks are used as prompts to guide SAM2 in generating potentially improved segmentations for the evaluation set.


## NOTE

Large Language Models were used generally to debug our code. 
