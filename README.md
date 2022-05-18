# vis_assignment1

## Assignment 1 description
In the last few weeks, we've seen how images can be deconstructed into the three colour channels which they comprise (RGB). We saw that, in some ways, the colour histogram of an image is its "colour fingerprint". We also saw that images can be compared for similarity of their colour histograms, allowing us to find which images are most like each other in terms of their colour.

For this assignment, you will write a small Python program to compare image histograms quantitively using Open-CV and the other image processing tools you've already encountered. Your script should do the following:

    Take a user-defined image from the folder
    Calculate the "distance" between the colour histogram of that image and all of the others.
    Find which 3 image are most "similar" to the target image.
    Save an image which shows the target image, the three most similar, and the calculated distance score.
    Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

## Methods
First, to get this to work you should download this dataset and put it in the ```in``` folder.
This problem dealt mainly with basic image processing, such as normalisation and creating histograms and comparing them to each other. To tackle the problem, I made two scripts, one (```image_searching.py```) using fairly simple image processing techniques to find the three most similar images, the second (```nearest_neighbours.py```) using feature extraction and nearest neighbour calculations instead.

image_searching.py iterates over the flowers folder to get the filepaths, does aforementioned image processing and finds the three closest images by comparing the distance scores of histograms. The user is able to define the focus image with the ```image``` argument. This will get the job done, but the second script is more sophisticated and generally gets better results.

nearest_neighbours.py also iterates over the flowers folder to get a list of features and to make a list of filepaths. It extracts the features by using a function we defined in class. Then it defines a pretrained model, in this case VGG16, which uses imagenet weights as a default. In the script, the user is able to define a ``` focus_index```. As the images in the flowers folder are called something like ```image_0001.jpg``` it is easy to index, as the index is part of the name. The output in the ```out``` folder is from index ```[531]```. 
Finally, it saves the focus image along with the three closest images and their distance scores. The user can define the name of the output file by using the ```vis_name``` argument.

## Usage
The first script, ```image_searching.py``` is very simple to use; simply direct the command line to the vis_assignment1 folder, run the script and include the required argument -i "*insert image path here*"

The second script, ```nearest_neigbours.py``` is also quite simple, but has more arguments. Again, direct the command line the vis_assignment1 folder, run the script and include the required arguments -fi "*insert focus image path here*" and -vn "*insert desired visualisation name here*". 

## Results
```image_searching.py``` generally does what it needs to do. It finds the three closest images and their names + distance scores, along with a visualisation. However ```nearest_neighbours.py``` uses a more sophisticated method and I would therefore recommend using that one, as the results are more reliable. ```nearest_neighbours.py``` does require quite a lot of computational power as it utilizes a pretrained model, so if you lack computational resources ```image_searching.py``` still works pretty well.


