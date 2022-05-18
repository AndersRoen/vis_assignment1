import os
import cv2
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_image_paths():
    # set a directory path
    directory_path = os.path.join("in", "flowers")
    # get the file names
    filenames = os.listdir(directory_path) 
    # create an empty list
    joined_paths = []
    # for every file in filenames
    for file in filenames:
        # if the file doesn't end with .jpg 
        if not file.endswith(".jpg"): # this is to avoid appending the added file thingy
            # do not append
            pass
        # if the file does end with .jpg
        else:
            # get directory path and the file name
            input_path = os.path.join(directory_path, file)
            # and append to the list
            joined_paths.append(input_path)
    # sort the list in ascending order        
    joined_paths = sorted(joined_paths)
    return joined_paths 

def hist_norm(image):
    # get the image
    img_focus = os.path.join("..", "CDS-VIS", "flowers", image)
    # extract the histogram
    hist = cv2.calcHist([cv2.imread(img_focus)],
                         [0,1,2],  
                         None, 
                         [8,8,8],  
                         [0,256, 0,256, 0,256])
    # normalize
    ready_img_focus = cv2.normalize(hist, hist, 0,255, cv2.NORM_MINMAX)
    return ready_img_focus

# do the same for all other images
def hist_norm_comp(joined_paths, ready_img_focus):
    # create an empty list
    ready_images = []
    # create another
    comp_images = []
    # for every file in joined_paths
    for file in joined_paths:
        # calculate the histograms
        image_hist = cv2.calcHist([cv2.imread(file)], [0, 1, 2], None, [8, 8, 8], [0,256, 0,256, 0,256])
        # normalize
        image_norm = cv2.normalize(image_hist, image_hist, 0,255, cv2.NORM_MINMAX)
        # and append to the list
        ready_images.append(image_norm)
    # for every file in ready_images    
    for image in ready_images:
        # compare the histograms with the user-defined images and get the distances
        image_comp = cv2.compareHist(ready_img_focus, image, cv2.HISTCMP_CHISQR)
        # and append the distances to the list
        comp_images.append(image_comp)
    # sort the list    
    compare_sort = sorted(comp_images)
    # find the most similar 
    similar_images = compare_sort[1:10]
    # get their scores
    img1 = similar_images[0]
    img2 = similar_images[1]
    img3 = similar_images[2]
    # create another empty list
    similar = []
    # get the indices of each of the distance scores from the unsorted list of distances scores
    image1 = comp_images.index(img1)
    image2 = comp_images.index(img2)
    image3 = comp_images.index(img3)
    # use this index to find the images in joined_paths
    join_image1 = joined_paths[image1]
    join_image2 = joined_paths[image2]
    join_image3 = joined_paths[image3]
    # and append the most similar images to the list
    similar.append([join_image1, join_image2, join_image3])
    return similar, similar_images

# create a function that saves the images with their distances scores and a csv with the filenames of the images
def save_images_csv(similar, joined_paths, image, similar_images):
    final_images = []
    # for every token in similar
    for o in similar:
        # for every element in the token
        for i in o:
            # read files as images
            sim_img = cv2.imread(os.path.join(i))
            # convert their colors to allow for plotting
            rgb_image = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
            # append to the list
            final_images.append(rgb_image)
    # do the same for the focus image
    rgb_focus = cv2.imread(os.path.join("..", "CDS-VIS", "flowers", image))
    rgb_focus = cv2.cvtColor(rgb_focus, cv2.COLOR_BGR2RGB)
    
    # let's plot the images
    # create a matrix of 2x2
    f, axarr = plt.subplots(2,2)
    
    # arrange each image in the matrix
    axarr[0,0].imshow(rgb_focus)
    axarr[0,1].imshow(final_images[0])
    axarr[1,0].imshow(final_images[1])
    axarr[1,1].imshow(final_images[2])
    
    # add the distances scores as text
    axarr[0,1].text(0.5, 0.5, f"Distance:{similar_images[0]}", fontsize=7, ha="center")
    axarr[1,0].text(0.5, 0.5, f"Distance:{similar_images[1]}", fontsize=7, ha="center")
    axarr[1,1].text(0.5, 0.5, f"Distance:{similar_images[2]}", fontsize=7, ha="center")
    
    # create an outpath
    outpath = os.path.join("Outputs", "color_hist_similar_images.jpg")
    
    # save the figure
    f.savefig(outpath)
    
    # use regex to clean the file paths in similar to just include the image names
    similar = [[re.sub(".*s\/", "", i) for i in token] for token in similar]
    
    # create a dictionary of what should go into the data frame
    data = [{"Original Image": image, "Third": similar[0][2], "Second": similar[0][1], "First": similar[0][0]}]
    
    # convert the dictionary to a data frame
    dframe = pd.DataFrame(data)
    
    # create another outpath
    outpath2 = os.path.join("Outputs", "color_hist_similar_images.csv")
    
    # and save the data frame as a csv
    dframe.to_csv(outpath2, encoding = "utf-8")
    
def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-i", "--image", required=True, help="the user-defined image")
    args = vars(ap.parse_args())
    return args

# run it
def main():
    args = parse_args()
    joined_paths = load_image_paths()
    ready_img_focus = hist_norm(args["image"])
    similar, similar_images = hist_norm_comp(joined_paths, ready_img_focus)
    save_images_csv(similar, joined_paths, args["image"], similar_images)
    
if __name__ == "__main__":
    main()
