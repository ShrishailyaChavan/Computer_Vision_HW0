# HW0: Alohomora
## Probability based edge detection

The homework is divided in two phases listed below. 
Click [here](https://rbe549.github.io/spring2023/hw/hw0/)  to view assignment requirements.
## Phase 1
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


