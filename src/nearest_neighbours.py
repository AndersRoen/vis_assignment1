import os 
import argparse
# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors

# for plotting
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt   
import matplotlib.image as mpimg


# defining the feature extraction function we'll use for the image search

def extract_features(img_path, model):
    # Define input image shape - remember to reshape
    input_shape = (224, 224, 3)
    img = load_img(img_path, target_size = (input_shape[0],
                                          input_shape[1]))
     # convert to array
    img_array = img_to_array(img)
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocess image - see last week's notebook
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    # flatten
    flattened_features = features.flatten()
    # normalise features
    normalized_features = flattened_features / norm(features)
    return normalized_features

# load in VGG16
def load_VGG16():
    model = VGG16(weights = "imagenet", # making the default parameter explicit
                  pooling = "avg",
                  include_top = False,
                  input_shape = (224, 224, 3))
    return model

# load in and iterate over folder
def load_iterate(model):
    directory_path = os.path.join("in", "flowers") # set a directory path
    filenames = os.listdir(directory_path) # list the contents of the directory in order

    joined_paths = []
    for file in filenames:
        if not file.endswith(".jpg"):
            pass
        else:
            input_path = os.path.join(directory_path, file)
            joined_paths.append(input_path)

    joined_paths = sorted(joined_paths)
    
    # feature extraction
    feature_list = [] # list of one entry of every embedding we have
    
    for input_file in joined_paths:
        features = extract_features(input_file, model)
        feature_list.append(features)
    
    return feature_list, joined_paths

# do the nearest neighbours and save the closest images
def n_neighbours(feature_list, focus_index, joined_paths, vis_name):
    neighbors = NearestNeighbors(n_neighbors = 10, # find the 10 nearest neighbors 
                                 algorithm = "brute",
                                 metric = "cosine").fit(feature_list) # fit our features to out k-nearest neighbors algorithm
    distances, indices = neighbors.kneighbors([feature_list[int(focus_index)]]) # this finds the closest images and returns the indices of those images
    
    # save indices
    idxs = []
    
    for i in range(1, 6):
        idxs.append(indices[0][i])
        
    # this saves the 3 closest images with their calculated distance scores
    f, ax = plt.subplots(2, 2)
    ax[0,0].imshow(mpimg.imread(joined_paths[int(focus_index)]))
    ax[0,1].imshow(mpimg.imread(joined_paths[idxs[0]]))
    ax[1,0].imshow(mpimg.imread(joined_paths[idxs[1]]))
    ax[1,1].imshow(mpimg.imread(joined_paths[idxs[2]]))
    ax[0,1].text(0.5, 0.5, f"Distance:{distances[0][1]}", fontsize=7, ha="center")
    ax[1,0].text(0.5, 0.5, f"Distance:{distances[0][2]}", fontsize=7, ha="center")
    ax[1,1].text(0.5, 0.5, f"Distance:{distances[0][3]}", fontsize=7, ha="center")
    outpath = os.path.join("Outputs", vis_name)
    f.savefig(outpath)
    
    return idxs

def parse_args():
    #initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-fi", "--focus_index", required = True, help = "the index of the image you want to focus on")
    ap.add_argument("-vn", "--vis_name", required = True, help = "the name of the visualisation")
    args = vars(ap.parse_args())
    return args

def main():
    args = parse_args()
    model = load_VGG16()
    feature_list, joined_paths = load_iterate(model)
    idxs = n_neighbours(feature_list, args["focus_index"], joined_paths, args["vis_name"])
    
if __name__ == "__main__":
    main()
    


