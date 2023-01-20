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
