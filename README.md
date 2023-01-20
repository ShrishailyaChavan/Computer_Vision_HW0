# HW0: Alohomora

The homework is divided in two phases listed below. 
Click [here](https://rbe549.github.io/spring2023/hw/hw0/)  to view assignment requirements.
## Phase 1 (Shake My Boundary)
In this homework, I developed a simplified version of pb, which finds boundaries by examining brightness, color, and texture information across multiple scales (different sizes of objects/image). The output of the  algorithm is a per-pixel probability of boundary. Used Berkeley Segmentation Data Set 500 (BSDS500). This dataset is available in folder  schavan_hw0/Phase1/BSDS500/.

This section is the implementation of the pb boundary detection algorithm introduced. It is somewhat different from the classical CV techniques that are universally used all over like in Sobel and Canny Filters it uses the texture and color information present in the image in addition to the intensity discontinuities as well.

This is done in 4 steps in the sections to follow: 
1) Filter Banks
2) Texture, Brightness and Color Maps T, B, C
3) Texture, Brightness and Color Gradients Tg, Bg, Cg
4) Pb-lite output combined with baselines

# Filter banks
Filter banks are a set of filters, that are applied over an image to extract multiple features. In this homework, filter banks have been used to extract the texture properties. The following subsections will describe in brief the implementation of DoG filters, Leung-Malik filters and Gabor filters.

![Sample Input Image](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/BSDS500/Images/1.jpg)

#### Oriented Derivative of Gaussian Filters
![DoG Filter](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/DOG.png)

#### Leung-Malik Filters
![LM Filter](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/LM.png)

#### Gabor Filters
![Gabor Filter](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Gabor.png)


# Texton Maps

The filters described in the previous section are to detect the texture properties in an image. Since each filter bank has multiple filters and three such filter banks have been used, the result is a vector of filter responses. This vector of filter response associated with each pixel is encoding a texture property. Based on this filter response vectors, pixels with similar texture property were clustered together using K-mean algorithm(K= 64). The output of K-mean clustering is a Texton map(τ )

![Texton Map](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Texton_Maps/Texton1.png)

# Brightness Maps
The image was clustered based on the brightness value for each pixel. The images were first converted to gray scale and K-mean algorithm(K=16) was used to get brightness maps.

![Brightness Map](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Brightness_Maps/Brightness_map1.png)

# Color Maps

The image consists of three color channals(RGB), describ- ing the color property at each pixel. The images have been clustered using the RGB value using K-mean algorithm(K=16) to obtain color maps.

![Color Map](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Color_Maps/Color1.png)


# Texture, Brightness and Color Gradients
To obtain Tg,Bg,Cg, we need to compute differences of values across different shapes and sizes. This can be achieved very efficiently by the use of Half-disc masks.

### Half disk masks
![Half Disk Mask](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/HDMasks.png)


### Texture Gradient
![Texture Gradient](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Texton_Gradients/Tg1.png)

### Brightness Gradient
![Brightness Gradient](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Brightness_Gradient/Bg1.png)

### Color Gradient
![Color Gradient](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Color_Gradients/Cg1.png)

# Sobel and Canny baseline
The outputs from Sobel and Canny edge detector are combined using weighted average method.

### Sobel baseline
![Sobel](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Solbel_Baseline/Sobel_1.png)

### Canny baseline
![Canny](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Canny_Baseline/Canny1.png)

### Pb-lite output
![Pb-Lite](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Code/Image_Outputs/Pb_lite%20Outputs/Pb1.png)


## Folder Content Structure
![Folder](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/HW0_Folder_Structure.png)

## Pre-requisite

 - Python3
 - Jupyter Notebook
 - VS Code
 - Cuda
 - Ubuntu

## Run the code

Change the location to

```sh
 {root_directory}/Phase1 
```
Run the following command
```sh
 python3 Code/Wrapper.py 
```

# Phase 2 (DEEP DIVE ON DEEP LEARNING)
In this section, a basic neural network and its modified version for classification on CIFAR10 dataset have been de- scribed. Later, a case study for ResNet, ResNext and DenseNet architecture was conducted. Refer [report](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Report.pdf) for more details.

### Dataset
CIFAR-10 is a dataset consisting of 60000, 32×32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. More details about the datset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

Sample images from each class of the CIFAR-10 dataset is shown below:
![Dataset](https://github.com/ShrishailyaChavan/Computer_Vision_HW0/blob/main/Phase2Dataset.png)


## Implementation
Trained a convolutional neural network on PyTorch for the task of classification. The input is a CIFAR-10 image and the output is the probabilities of 10 classes. Used appropriate Loss Function and Optimizer as hyperparameters to determine the training and testing accuracy the was performed over epochs. Tried to improve accuracy by standardizing, augmenting the data images. Using ResNET, ResNEXT, DenseNET architectures, tried to make efficient use in-terms of memory usage (number of parameters), computation (number of operations) and accuracy.

Implemented below networks:
 - Simple Network
 - Improved CNN Network
 - ResNET
 - ResNEXT
 - DenseNET
 


Sample test runs, Network files, testing, training code files can be found in Phase2/Code/ folder for each network that was implemented.

Analysis report is created and available in the source code that is submitted based on the results obtained.
 
 ## Pre-requisite

 - Python3
 - Jupyter Notebook
 - VS Code
 - Cuda
 - Ubuntu

## Run the code

Run the following command
```sh
 python3  {root_directory}/HW0Phase1AndPhase2Notebook.ipynb
```
 or
 
 to run a specific network, go to
 {root_directory}/Phase2/Code/Network/SimpleNetwork.py
 Also, run the Train.py and Test.py files.
 

