#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils
#from google.colab.patches import cv2_imshow


def gaussian_kernel_2D(sigma, kernel_size):

    #value = int((kernel_size-1)/2)
    g = np.zeros([kernel_size, kernel_size])
    
    if (kernel_size % 2) == 0:
        value = kernel_size / 2
    else:
        value = (kernel_size)/2

    x,y = np.meshgrid(np.linspace(-value,value,kernel_size), np.linspace(-value,value,kernel_size))
    
    #As there is is no sigma in y direction so it is equal to sigma_X
    sigma_x = sigma
    sigma_y = sigma

    exponential_part = np.exp((-1/2) * (np.square(x) /np.square(sigma_x) + np.square(y) / np.square(sigma_y)))

    normal = (1 / np.sqrt(2 * np.pi * sigma_x * sigma_y))

    kernel_value = exponential_part * normal
    
    kernel_2D = np.reshape(kernel_value, (kernel_size, kernel_size))
    return kernel_2D
    
    


def DoGfilter(scales, orientation, kernel_size):
    filter_bank = []

    #SOBEL Kernals in X and Y direction respectively
    Sx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])
    orientations = np.linspace(0,360,orientation)
    for scale in scales:
        
        Gaussian = gaussian_kernel_2D(scale, kernel_size)
        sigma = [scale, scale]
        Gaussian_x = cv2.filter2D(Gaussian,-1, Sx)
        Gaussian_y = cv2.filter2D(Gaussian,-1, Sy)
        for i in enumerate(orientations):
            filters = (Gaussian_x * np.cos(i[1])) +  (Gaussian_y * np.sin(i[1]))
            filter_bank.append(filters)
    
    return filter_bank


def Gaussian_Kernel_2d(sigma, kernel_size):	

	sigma_x, sigma_y = sigma

	Gaussian = np.zeros([kernel_size, kernel_size])
    
    #defining the index value according to kernel size
	if (kernel_size%2) == 0:
		index = kernel_size/2
	else:
		index = (kernel_size - 1)/2

	x, y = np.meshgrid(np.linspace(-index, index, kernel_size), np.linspace(-index, index, kernel_size))
 
	exponential_part = (np.square(x)/np.square(sigma_x)) + (np.square(y)/np.square(sigma_y))
 
	exponential_part = exponential_part / 2
	Gaussian = (0.5/(np.pi * sigma_x * sigma_y)) * np.exp(-exponential_part)
	return Gaussian

def DoGfilter(scales, orientation, kernel_size):
    filter_bank = []

    #SOBEL Kernals in X and Y direction respectively
    Sx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])
    orientations = np.linspace(0,360,orientation)
    for scale in scales:
        
        Gaussian = gaussian_kernel_2D(scale, kernel_size)
        sigma = [scale, scale]
        Gaussian_x = cv2.filter2D(Gaussian,-1, Sx)
        Gaussian_y = cv2.filter2D(Gaussian,-1, Sy)
        for i in enumerate(orientations):
            filters = (Gaussian_x * np.cos(i[1])) +  (Gaussian_y * np.sin(i[1]))
            filter_bank.append(filters)
    
    return filter_bank


def save_file(filter_name, file_directory, columns):

    #calculating the number of rows
    no_of_rows = int((len(filter_name)/ columns) + ((len(filter_name) % columns) != 0));
    plt.subplots(no_of_rows, columns, figsize=(15,15))
    #iterating through every image to plot the image
    for index in range(len(filter_name)):
        plt.subplot(no_of_rows, columns, index+1)
        #turning off the axis
        plt.axis('off')
        plt.imshow(filter_name[index], cmap='gray')
    plt.savefig(file_directory)
    plt.close()
dog_filters = DoGfilter(scales=[3,4], orientation = 16, kernel_size = 81)
save_file(dog_filters,"/home/DoG.png",16)

def LM_Filters(scales, orientations, filter):

	scale_1 = scales[0:3]
	gaussian_scale = scales
	Laplacian_of_Gaussian_scale = scales + [i * 3 for i in scales]

	filter_bank = []
	gaussian_1d = []
	gaussian_2d = []
	gaussian = []
	Laplacian_of_Gaussian = []
	

	kernel_x = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	kernel_y = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]])

	for scale in scale_1:
		sigma = [3*scale, scale]
		Gaussian = Gaussian_Kernel_2d(sigma, filter)
		

        #Gaussian 1-D filter 
		Gaussian_1d = cv2.filter2D(Gaussian, -1, kernel_x) + cv2.filter2D(Gaussian, -1, kernel_y)
        
        
        #Gaussian 2-D filter
		Gaussian_2d = cv2.filter2D(Gaussian_1d, -1, kernel_x) + cv2.filter2D(Gaussian_1d, -1, kernel_y)

		for orientation in range(orientations):
			filter_orientation = orientation * 180 / orientations

            #rotation using imutils function
			gaussian1D =  imutils.rotate(Gaussian_1d, filter_orientation)
			gaussian_1d.append(gaussian1D)

			gaussian2D = imutils.rotate(Gaussian_2d, filter_orientation)
			gaussian_2d.append(gaussian2D)
	
	
	for scale in Laplacian_of_Gaussian_scale:
		sigma = [scale, scale]
		Gaussian = Gaussian_Kernel_2d(sigma, filter)
		Laplacian_of_Gaussian_kernal = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
		Laplacian_of_Gaussian.append(cv2.filter2D(Gaussian, -1, Laplacian_of_Gaussian_kernal))
		#LoG.append(convolve2d(G, log_kernal))



	for scale in gaussian_scale:
		sigma = [scale, scale]
		gaussian.append(Gaussian_Kernel_2d(sigma, filter))


	filter_bank = gaussian_1d + gaussian_2d + Laplacian_of_Gaussian + gaussian
	return filter_bank


def save_file(filter_name, file_directory, columns):

    #calculating the number of rows
    no_of_rows = int((len(filter_name)/ columns) + ((len(filter_name) % columns) != 0));
    plt.subplots(no_of_rows, columns, figsize=(15,15))
    #iterating through every image to plot the image
    for index in range(len(filter_name)):
        plt.subplot(no_of_rows, columns, index+1)
        #turning off the axis
        plt.axis('off')
        plt.imshow(filter_name[index], cmap='gray')
    plt.savefig(file_directory)
    plt.close()

LMS_filters = LM_Filters([1, np.sqrt(2), 2, 2*np.sqrt(2)], 6, 49)
save_file(LMS_filters,"/home/LM.png", 12)

#changing sigma values to chnaeg the for LML as it is different than LMS_filters

LML_filters = LM_Filters([np.sqrt(2), 2, 2*np.sqrt(2), 4], 6, 49)
save_file(LML_filters, "/home/LM.png", 12)
LM_filter_bank = LMS_filters + LML_filters
save_file(LM_filter_bank, "/home/LM.png", 12)


def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    #Lambda = wavelength of the sinusoidal factor
    #theta = orientation of the normal o the parallel waves
    #psi = phase effect
    #gamma = spatial aspect ratio

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

#As we know 2 - D Gabor Filter is a Gaussian kernel Function modulated by sinusoidal plane wave
def sin_wave(kernel_size, frequency, theta):
    if (kernel_size%2) == 0:
        index_value = kernel_size / 2
    else:
        index_value = (kernel_size - 1) / 2

    (x, y) = np.meshgrid(np.linspace(-index_value, index_value, kernel_size), np.linspace(-index_value, index_value, kernel_size))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    #total_theta = x * np.cos(theta) + y * np.sin(theta)
    total_theta = x_theta + y_theta


    sin_wave = np.sin(total_theta * 2 * np.pi * frequency/kernel_size)

    return sin_wave

#defining the gabor filter function
def gaborFilter(scales, orientations, frequencies, filter_size):
    filter_bank = []
    for scale in scales:
        sigma = [scale, scale]
        Gaussian = Gaussian_Kernel_2d(sigma, filter_size)
        for frequency in frequencies:
            for orientation in range(orientations):
                filter_orientation = orientation * np.pi / orientations
                sin2d = sin_wave(filter_size, frequency, filter_orientation)
                gabor_filter = Gaussian * sin2d
                filter_bank.append(gabor_filter)
    return filter_bank  

def save_file(filter_name, file_directory, columns):

    #calculating the number of rows
    no_of_rows = int((len(filter_name)/ columns) + ((len(filter_name) % columns) != 0));
    plt.subplots(no_of_rows, columns, figsize=(15,15))
    #iterating through every image to plot the image
    for index in range(len(filter_name)):
        plt.subplot(no_of_rows, columns, index+1)
        #turning off the axis
        plt.axis('off')
        plt.imshow(filter_name[index], cmap='gray')
        plt.savefig(file_directory)
        plt.close()
gabor_filter_bank = gaborFilter([8,16,24], 8, [2,4,6], 49)
save_file(gabor_filter_bank,"/home/Gabor.png",8)
image = cv2.imread("Phase1/Results/Gabor.png")
cv2.imshow(image)


def Half_Disk(radius, orientation):
    #size of the kernel must be 
    kernel_size = 2*radius 
    centre = radius 
    mask_size = np.zeros([kernel_size, kernel_size])
    for x in range(radius):
        for y in range(kernel_size):
            #calculating the distance from centre
            distance_from_centre = np.square(x-radius) + np.square(y-radius)

            if (distance_from_centre <= np.square(radius)):
                mask_size[x,y] = 1

    mask_size = imutils.rotate(mask_size, orientation)
    #rounding up the values of mask size
    #mask_size[mask_size<=0.5] = 0
    #mask_size[mask_size>0.5] = 1
    mask_size = np.round(mask_size) 
    
    return mask_size


def Half_Disk_FilterBank(radius, orientations):

    filter_bank_pair = []
    
    #setting up orientation for the asked orientations in question
    orientation=np.linspace(0,360,orientations)

    for radii in radius:
        for orient in orientation:
            half_mask = Half_Disk(radii,orient)
            halfmask_rotation = imutils.rotate(half_mask,180)

            #rounding up the rotation values
            halfmask_rotation = np.round(halfmask_rotation)
            filter_bank_pair.append(half_mask)
            filter_bank_pair.append(halfmask_rotation)

    return filter_bank_pair

def save_file(filter_name, file_directory, columns):

    #calculating the number of rows
    no_of_rows = int((len(filter_name)/ columns) + ((len(filter_name) % columns) != 0));
    plt.subplots(no_of_rows, columns, figsize=(15,15))
    #iterating through every image to plot the image
    for index in range(len(filter_name)):
        plt.subplot(no_of_rows, columns, index+1)
        #turning off the axis
        plt.axis('off')
        plt.imshow(filter_name[index], cmap='gray')
    plt.savefig(file_directory)
    plt.close()

halfdisk_filterbank = Half_Disk_FilterBank([4,6,8], 8)
save_file(halfdisk_filterbank,"/home/halfdisk.png",8)

image = cv2.imread('/home/halfdisk.png')
cv2.imshow(image)


from sklearn.cluster import KMeans
image = cv2.imread('Phase1/BSDS500/Images/1.jpg')
#converting the image to gray_scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def TextonMap(image, Derivative_of_Gaussian, Leung_Malik, Gabor):
    #converting image to an array form
    Image = np.array(image)

    for i in range(len(Derivative_of_Gaussian)):

        Convolution_of_image = cv2.filter2D(image,-1, Derivative_of_Gaussian[i])
        Image = np.dstack((Image,Convolution_of_image))
        
    for i in range(len(Leung_Malik)):
        Convolution_of_image = cv2.filter2D(image,-1, Leung_Malik[i])
        Image = np.dstack((Image,Convolution_of_image))
        
    for i in range(len(Gabor)):
        Convolution_of_image = cv2.filter2D(image,-1, Gabor[i])
        Image = np.dstack((Image,Convolution_of_image))
    
    Image = Image[:,:,1:]
    return Image
                

texton_map = TextonMap(image, dog_filters, LM_filter_bank, gabor_filter_bank)


image = cv2.imread('Phase1/BSDS500/Images/1.jpg')

def Texton(image, clusters):
    x,y,z = image.shape
    image = np.reshape(image,((x*y),z))
    kmeans_clustering = KMeans(n_clusters = clusters, random_state = 4)
    kmeans_clustering.fit(image)
    labels = kmeans_clustering.predict(image)
    reshape = np.reshape(labels,(x,y))

    return reshape
texton = Texton(image, 64)
cv2.imshow(texton)
cv2.imwrite('/home/texton.jpg', image)


from sklearn.cluster import KMeans
image = cv2.imread('/content/data/Images/1.jpg')
#converting the image to gray_scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.float32(image)

def Chi_Square_Distance(image, brightness_bins, filter_1, filter_2):
    
    chi_square_distance = []
    chi_square_dist = np.zeros(image.shape)
    #finding the minimum value of brightness bins
    minimum_brightness_bins = np.min(image)
    tmp = np.zeros(image.shape)
 

    for i in range(brightness_bins):
        #tmp = 1 where image is in bin i and 0 elsewhere
        tmp[image == i] = 1.0
        tmp[image != i] = 0.0

        #convolving our image with the respectivre filter values
        g_i = cv2.filter2D(tmp,-1,filter_1)
        h_i = cv2.filter2D(tmp,-1,filter_2)


        chi_square_dist = chi_square_dist + np.square(g_i - h_i)/(g_i + h_i + np.exp(-10))
        
        #Dividing the variable by 2 and assigns the result to that variable as asked in formula
        chi_square_dist = chi_square_dist / 2

        #chi_square_distance.append(chi_square_dist)

    return chi_square_dist 

def Gradient(image, brightness_bins, halfdisk_filterbank):
    #we are defining the Gradient function based on hald_disks to compare the distributions
    #index = len((halfdisk_filterbank)/2)
    for i in range(int(len(halfdisk_filterbank)/2)):
        # selecting the value of i from from half of the length of halfdisk_filter banl to avoid the index being out of range
        #creating two opposite directional filters 
        filter_1 = halfdisk_filterbank[i]
        filter_2 = halfdisk_filterbank[i+1]

        chisquare_distance = Chi_Square_Distance(image, brightness_bins, filter_1, filter_2)

        #chisquare_distance = np.array(chisquare_distance)

        #gradient = np.mean(chisquare_distance, axis=0)

        
        gradient = np.dstack((image,chisquare_distance))
        gradient = gradient[:,:,1:]

    return gradient

Tg = Gradient(texton, 64, halfdisk_filterbank)
Tg = Tg[:,:,0]

plt.imshow(Tg)
cv2.imwrite('/home/Tg.jpg', Tg)


from sklearn.cluster import KMeans
image = cv2.imread('/content/data/Images/1.jpg')
#converting the image to gray_scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def Brightness_Map(image, clusters):

    x,y = image.shape
    image = np.reshape(image,((x*y),1))
    kmeans_clustering = KMeans(n_clusters = clusters, random_state = 4)
    kmeans_clustering.fit(image)
    labels = kmeans_clustering.predict(image)
    reshape = np.reshape(labels,(x,y))
    return reshape
brightness_map =Brightness_Map(image,16)

#normalzing the image
norm_image = cv2.normalize(brightness_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.imshow(norm_image)

cv2.imwrite('/home/Brightness_map.jpg', norm_image)

Brightness_Gradient = Gradient(brightness_map, 16, halfdisk_filterbank)
# Brightness_Gradient is a 3D matrix with 1 filter but we need a 2D matrix with first two dimensions of image 
Brightness_Gradient.shape

#Getting first two dimesnions to print an image and removing third dimension of filter
Brightness_Gradient = Brightness_Gradient[:,:,0]

plt.imshow(Brightness_Gradient)

cv2.imwrite('/home/Brightness_Gradient.jpg', Brightness_Gradient)


from sklearn.cluster import KMeans
image = cv2.imread('/content/data/Images/1.jpg')
#converting the image to gray_scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def Color_Map(image, clusters):
    a,b = image.shape
    image = np.reshape(image,((a*b),1))
    kmeans = KMeans(n_clusters = clusters, random_state = 4)
    kmeans.fit(image)
    labels = kmeans.predict(image)
    print(labels.shape)
    reshape = np.reshape(labels,(a,b))
    print(reshape.shape)
    return reshape
color_map = Color_Map(image,16)

norm_image = cv2.normalize(color_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.imshow(norm_image)

cv2.imwrite('/home/Color_Map.jpg', norm_image)

Color_gradient = Gradient(color_map, 16, halfdisk_filterbank)
Color_gradient.shape
# Color_Gradient is a 3D matrix with 1 filter but we need a 2D matrix with first two dimensions of image

#Getting first two dimesnions to print an image and removing third dimension of filter
Color_gradient = Color_gradient[:,:,0]

plt.imshow(Color_gradient)

cv2.imwrite('/home/Color_gradient.jpg', Color_gradient)




def main():



	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

dog_filters = DoGfilter(scales=[3,4], orientation = 16, kernel_size = 81)
save_file(dog_filters,"Phase1/Results/DoG.png",16)
image = cv2.imread('Phase1/Results/DoG.png')
cv2.imshow(image)
	
cv2.imwrite('Phase1/Results/DoG.png', image)


"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
"""

LMS_filters = LM_Filters([1, np.sqrt(2), 2, 2*np.sqrt(2)], 6, 49)
save_file(LMS_filters,"/home/LM.png", 12)

    #changing sigma values to chnaeg the for LML as it is different than LMS_filters

LML_filters = LM_Filters([np.sqrt(2), 2, 2*np.sqrt(2), 4], 6, 49)
save_file(LML_filters, "/home/LM.png", 12)
LM_filter_bank = LMS_filters + LML_filters
save_file(LM_filter_bank, "/home/LM.png", 12)

"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
"""

gabor_filter_bank = gaborFilter([8,16,24], 8, [2,4,6], 49)
save_file(gabor_filter_bank,"/home/Gabor.png",8)
image = cv2.imread("Phase1/Results/Gabor.png")
cv2.imshow(image)

"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
"""

halfdisk_filterbank = Half_Disk_FilterBank([4,6,8], 8)
save_file(halfdisk_filterbank,"/home/halfdisk.png",8)

image = cv2.imread('/home/halfdisk.png')
cv2.imshow(image)


"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
"""
from sklearn.cluster import KMeans
image = cv2.imread('/content/data/Images/1.jpg')
 #converting the image to gray_scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def TextonMap(image, Derivative_of_Gaussian, Leung_Malik, Gabor):
    #converting image to an array form
    Image = np.array(image)

    for i in range(len(Derivative_of_Gaussian)):

        Convolution_of_image = cv2.filter2D(image,-1, Derivative_of_Gaussian[i])
        Image = np.dstack((Image,Convolution_of_image))
        
    for i in range(len(Leung_Malik)):
        Convolution_of_image = cv2.filter2D(image,-1, Leung_Malik[i])
        Image = np.dstack((Image,Convolution_of_image))
        
    for i in range(len(Gabor)):
        Convolution_of_image = cv2.filter2D(image,-1, Gabor[i])
        Image = np.dstack((Image,Convolution_of_image))
    
        Image = Image[:,:,1:]
        return Image

texton_map = TextonMap(image, dog_filters, LM_filter_bank, gabor_filter_bank)

"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
"""
image = cv2.imread('/content/data/Images/1.jpg')
def Texton(image, clusters):
    x,y,z = image.shape
    image = np.reshape(image,((x*y),z))
    kmeans_clustering = KMeans(n_clusters = clusters, random_state = 4)
    kmeans_clustering.fit(image)
    labels = kmeans_clustering.predict(image)
    reshape = np.reshape(labels,(x,y))

    return reshape

texton = Texton(image, 64)
cv2.imshow(texton)
cv2.imwrite('Phase1/Results/texton.jpg', image)


"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
"""
from sklearn.cluster import KMeans
image = cv2.imread('/content/data/Images/1.jpg')
#converting the image to gray_scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.float32(image)

def Chi_Square_Distance(image, brightness_bins, filter_1, filter_2):
    
    chi_square_distance = []
    chi_square_dist = np.zeros(image.shape)
   	 #finding the minimum value of brightness bins
    minimum_brightness_bins = np.min(image)
    tmp = np.zeros(image.shape)
 

    for i in range(brightness_bins):
       	#tmp = 1 where image is in bin i and 0 elsewhere
        tmp[image == i] = 1.0
        tmp[image != i] = 0.0

        #convolving our image with the respectivre filter values
        g_i = cv2.filter2D(tmp,-1,filter_1)
        h_i = cv2.filter2D(tmp,-1,filter_2)


        chi_square_dist = chi_square_dist + np.square(g_i - h_i)/(g_i + h_i + np.exp(-10))
        
        #Dividing the variable by 2 and assigns the result to that variable as asked in formula
        chi_square_dist = chi_square_dist / 2

        #chi_square_distance.append(chi_square_dist)

        return chi_square_dist 



"""
	Generate Brightness Map
	Perform brightness binning 
"""
from sklearn.cluster import KMeans
image = cv2.imread('/content/data/Images/1.jpg')
#converting the image to gray_scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def Brightness_Map(image, clusters):

    x,y = image.shape
    image = np.reshape(image,((x*y),1))
    kmeans_clustering = KMeans(n_clusters = clusters, random_state = 4)
    kmeans_clustering.fit(image)
    labels = kmeans_clustering.predict(image)
    reshape = np.reshape(labels,(x,y))
		
    return reshape
brightness_map =Brightness_Map(image,16)

	#normalzing the image
norm_image = cv2.normalize(brightness_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.imshow(norm_image)
cv2.imwrite('/home/Brightness_map.jpg', norm_image)



"""
Generate Brightness Gradient (Bg)
Perform Chi-square calculation on Brightness Map
Display Bg and save image as Bg_ImageName.png,
use command "cv2.imwrite(...)"
"""
Brightness_Gradient = Gradient(brightness_map, 16, halfdisk_filterbank)
# Brightness_Gradient is a 3D matrix with 1 filter but we need a 2D matrix with first two dimensions of image 
Brightness_Gradient.shape

#Getting first two dimesnions to print an image and removing third dimension of filter
Brightness_Gradient = Brightness_Gradient[:,:,0]

plt.imshow(Brightness_Gradient)

cv2.imwrite('/home/Brightness_Gradient.jpg', Brightness_Gradient)



"""
Generate Color Map
Perform color binning or clustering
"""
from sklearn.cluster import KMeans
image = cv2.imread('/content/data/Images/1.jpg')
#converting the image to gray_scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def Color_Map(image, clusters):
    a,b = image.shape
    image = np.reshape(image,((a*b),1))
    kmeans = KMeans(n_clusters = clusters, random_state = 4)
    kmeans.fit(image)
    labels = kmeans.predict(image)
    print(labels.shape)
    reshape = np.reshape(labels,(a,b))
    print(reshape.shape)
    return reshape
color_map = Color_Map(image,16)

#NORMALIZING THE IMAGE
norm_image = cv2.normalize(color_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.imshow(norm_image)


"""
Generate Color Gradient (Cg)
Perform Chi-square calculation on Color Map
Display Cg and save image as Cg_ImageName.png,
use command "cv2.imwrite(...)"
"""

Color_gradient = Gradient(color_map, 16, halfdisk_filterbank)
Color_gradient.shape
# Color_Gradient is a 3D matrix with 1 filter but we need a 2D matrix with first two dimensions of image

#Getting first two dimesnions to print an image and removing third dimension of filter
Color_gradient = Color_gradient[:,:,0]

plt.imshow(Color_gradient)

cv2.imwrite('/home/Color_gradient.jpg', Color_gradient)


"""
Read Sobel Baseline
use command "cv2.imread(...)"
"""

Sobel_Baseline = cv2.imread('/content/data/BSDS500/SobelBaseline/1.png')
cv2.imshow(Sobel_Baseline)

"""
Read Canny Baseline
use command "cv2.imread(...)"
"""
Canny_Baseline = cv2.imread("/content/data/BSDS500/CannyBaseline/1.png")
cv2.imshow(Canny_Baseline)
"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
"""
Average_of_gradient = (Tg + Brightness_Gradient + Color_gradient) / 3

w1 = 0.5
w2 = 0.5

canny_pb = Canny_Baseline
sobel_pb = Sobel_Baseline

a = (w1 * canny_pb + w2 * sobel_pb)
a = a[:,:,0]
Pb_lite_output = (Average_of_gradient) * a




if __name__ == '__main__':
    main()
 


