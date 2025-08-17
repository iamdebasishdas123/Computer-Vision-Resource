# Computer Vision Lession Plan

## Basics of Computer Vision
- Best Video: [Campusx Computer Vision Playlist[40-53]](https://youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn&si=7PxPHwRoU6Nm82dZ)

- Best Blog: [Basics of Computer Vision](https://medium.com/@iamdebasishdas123/basics-of-cnn-1bd993493d6b)

Demo Code: [Google Colab]()
## Image Processing Techniques

### 1. Filtering and Convolution

#### Gaussian Blur

**Definition:**  
Gaussian Blur is a widely used image processing technique that smooths an image by averaging the pixels within a region, using a weighted kernel derived from the Gaussian function. This reduces image detail and noise, resulting in a soft, blurry appearance.

**How It Works:**  
- A Gaussian kernel (a matrix of weights following the shape of a Gaussian curve) is created.
- The kernel is moved (convolved) over each pixel in the image.
- For each position, the new pixel value is calculated as a weighted average of the neighboring pixels, with the center pixel and those closer to the center given higher weights.
- The process blurs edges and reduces detail according to the size (standard deviation) of the Gaussian kernel.

**Use Cases:**
- **Noise Reduction:** Smooths out grainy images or sensor noise before further analysis.
- **Preprocessing:** Prepares images for edge detection or feature extraction by removing high-frequency components.
- **Background Blurring:** Used for artistic effect or in computer vision pipelines to separate objects from a less-detailed background.
- **Reducing Detail:** Softens facial features or blemishes in portrait editing.
- **Anti-aliasing:** Prevents pixelation effects in graphics and text rendering.

- Best Video: [What is Gaussian Blur?]()

- Blog: [What is Gaussian Blur?](https://aryamansharda.medium.com/image-filters-gaussian-blur-eb36db6781b1)


#### Sobel Operator

**Definition:**  
The Sobel Operator is an edge detection technique used in image processing and computer vision. It calculates the gradient magnitude and direction at each pixel, highlighting regions with high spatial frequency that typically correspond to edges in the image.

**How It Works:**  
- The Sobel operator uses two 3x3 convolution kernels: one for detecting changes in the horizontal direction (Sobel-X) and one for the vertical direction (Sobel-Y).
- Each kernel slides over the image, computing weighted sums of the neighboring pixel values.
- The result is two gradient images (Gx and Gy), representing the rate of change in intensity along the x (horizontal) and y (vertical) axes.
- The overall edge strength (gradient magnitude) at each pixel is usually calculated as:  
  \[
  Gradient Magnitude = sqrt [Gx^2 + Gy^2]
  \]
- The direction (angle) of the edge can also be found using the arctangent of Gy/Gx.
- Strong values correspond to edges; low values correspond to flat regions.

**Use Cases:**
- **Edge Detection:** Identifies object boundaries and sharp transitions in images.
- **Feature Extraction:** Provides information for tasks like image segmentation and corner detection.
- **Preprocessing for Computer Vision:** Serves as an initial step for more complex image analysis pipelines, including object recognition and tracking.
- **Medical Imaging:** Highlights structures such as blood vessels or cell boundaries.
- **Document Scanning:** Detects and enhances edges in scanned documents or handwriting.

- Best Video: [How To Find Edges In Images: Sobel Operators & Full Implementation](https://www.youtube.com/watch?v=VL8PuOPjVjY)

- Best Blog: [ Sobel Operator - 1](https://aryamansharda.medium.com/how-image-edge-detection-works-b759baac01e2)

- Another Blog: [ Sobel Operator](https://medium.com/@erhan_arslan/exploring-edge-detection-in-python-2-sobel-edge-detector-a-closer-look-de051a7b56df)

Demo code: [Goggle Colab]()

### 2. Morphological Operations

**Definition:**  
Morphological operations are a set of image processing techniques that process images based on their shapes. They apply a structuring element to an input image, usually in binary (black & white) form, to alter its geometry—such as shrinking, expanding, or cleaning up objects in the image.

**How It Works:**  
- A small matrix called a **structuring element** is moved across the image.
- At each location, the structuring element compares its shape to the underlying pixels, modifying them according to the chosen operation.
- Common basic operations:
  - **Erosion:** Shrinks white (foreground) regions and enlarges dark (background) areas by removing pixels around object boundaries.
  - **Dilation:** Expands white (foreground) regions, growing objects and shrinking holes/gaps.
  - **Opening:** Erosion followed by dilation; removes small objects/noise while preserving main shapes.
  - **Closing:** Dilation followed by erosion; fills small holes and gaps within objects while retaining shape.
- Variants and combinations of these operations can extract or modify more complex structures.

**Use Cases:**
- **Noise Removal:** Cleans up isolated pixels or small artifacts in binary images.
- **Shape Extraction:** Highlights or isolates particular features of interest (lines, boundaries, etc.).
- **Object Separation:** Splits touching or overlapping objects for counting or measurement.
- **Hole Filling:** Fills in small gaps within objects.
- **Image Preprocessing:** Prepares images for further processing, such as contour detection or skeletonization.

- Best Video: [ Morphological Operations | Erosion and Dilation  ](https://www.youtube.com/watch?v=r8ocf43NyQA&ab_channel=SHUBHAMARORA)

- Best Blog: [OpenCV: Morphological Dilation and Erosion](https://medium.com/@sasasulakshi/opencv-morphological-dilation-and-erosion-fab65c29efb3)

- Best Blog: [Erosion (Morphological Operation) — Image Processing](https://medium.com/@anshul16/erosion-morphological-operation-image-processing-18537f7c66cd)

Demo Code: [](https://github.com)

### 3. Histogram Equalization

- Best Video: [Histogram Equalization Image Enhancement Technique](https://www.youtube.com/watch?v=cVg2WiAX8Lg)

- Another Video: [Histogram Equalization](https://youtu.be/tn2kmbUVK50?si=l6B3WKejjEVwj9fA)

- Best Blog: [Medium Blog Histogram Eaqulization ](https://medium.com/@kyawsawhtoon/a-tutorial-to-histogram-equalization-497600f270e2)

Demo Code: [Google Colab]()

## Feature Detection and Description

### 1. SIFT (Scale-Invariant Feature Transform)

**Definition:**  
SIFT is an advanced algorithm for detecting and describing local features in images. It identifies distinctive, invariant keypoints that can be reliably matched across images, even if they are rotated, scaled, or slightly changed in viewpoint or illumination.

**How It Works:**  
- The algorithm identifies keypoints by searching for maxima and minima in the difference-of-Gaussians across multiple scales (image pyramid).
- For each keypoint, SIFT computes a descriptor by analyzing the gradient orientations in its local neighborhood, resulting in a unique, robust feature vector.
- The keypoints and their descriptors are designed to be invariant to scale, rotation, and moderately robust to changes in affine distortion and noise.
- Matching is typically performed by comparing SIFT descriptors using distance metrics such as Euclidean distance.

**Use Cases:**
- **Image Stitching:** Aligns and blends multiple images for panorama creation.
- **Object Recognition:** Detects and identifies objects in different scenes and under various viewing conditions.
- **Robust Image Matching:** Finds correspondences between images taken from different angles, scales, or lighting.
- **3D Reconstruction:** Establishes feature matches across multiple views for building 3D scene geometry.
- **Robot Navigation:** Provides stable landmarks for robots to recognize and localize themselves in an environment.

- Best Video: [ SIFT Introduction | Scale Invariant Feature Transform | Computer Vision](https://www.youtube.com/watch?v=ttD3pvM6pEI)

- Another Video: [SIFT Mathematics](https://youtube.com/playlist?list=PLlCkKK04bmVlvCs-S-2DnGf08MY2Hdd0n&si=Zrb7eK0wfGxpHKLc)

- Best Blog: [Medium Blog](https://medium.com/@deepanshut041/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40)

Demo Code: []()

### 2. SURF (Speeded-Up Robust Features)

**Definition:**  
SURF is a feature detection and description algorithm designed to quickly identify and describe distinctive, repeatable keypoints in images. Like SIFT, SURF provides robustness to scale, rotation, and moderate viewpoint or illumination changes, but it is optimized for increased speed.

**How It Works:**  
- SURF detects keypoints using a fast approximation of the Hessian matrix, which identifies salient regions of the image efficiently at multiple scales.
- It uses box filters and integral images to accelerate computation, making it significantly faster than SIFT.
- For each detected keypoint, SURF computes a descriptor based on the distribution of intensity changes (gradients) within a local neighborhood, resulting in a distinctive and compact feature vector.
- These descriptors are used to robustly match features between images, even under changes in scale, rotation, or moderate affine distortion.

**Use Cases:**
- **Real-Time Object Recognition:** Suitable for fast applications where speed matters, such as augmented reality and robotics.
- **Image Registration:** Aligns images taken at different times or from different viewpoints.
- **Tracking:** Follows features across video frames for motion estimation or visual odometry.
- **Image Retrieval:** Helps in finding duplicate or similar images from large databases efficiently.
- **Panorama Stitching:** Combines overlapping images by matching features quickly and reliably.

- Best Video: [SURF](https://youtu.be/PBTrwymDVCg?si=Xf2diGT_z8rEBW-q)

- Best Blog: [Medium](https://medium.com/@deepanshut041/introduction-to-surf-speeded-up-robust-features-c7396d6e7c4e)

Demo Code: []()

### 3. HOG Feature Descriptor (Histogram Of Oriented Gradient)

**Definition:**  
HOG is a feature descriptor used to capture the structure or the appearance of an object within an image. It is especially effective for object detection tasks, such as pedestrian or vehicle detection, by characterizing local object shape through the distribution of gradient orientations.

**How It Works:**  
- The image is divided into small, equally sized regions called cells.
- For each cell, the algorithm computes the gradient magnitude and orientation of each pixel.
- It then creates a histogram of gradient orientations within each cell.
- Cells are grouped into larger blocks to normalize the histograms, increasing robustness to changes in illumination and contrast.
- The normalized histograms from all blocks are concatenated to form the HOG descriptor for the image or region.
- This final descriptor can be used to train classifiers (like SVM) for detection tasks.

**Use Cases:**
- **Pedestrian Detection:** Widely used for detecting people in images and videos.
- **Vehicle Detection:** Helps in identifying cars, bicycles, and other vehicles.
- **Object Localization:** Locates and classifies various object types in computer vision pipelines.
- **Image Retrieval:** Finds similar patterns or shapes based on texture and edge orientations.
- **Human-Computer Interaction:** Facilitates gesture and activity recognition using full-body or hand silhouettes.

- Best Video: [ Hog ](https://www.youtube.com/watch?v=Z2ml7WzCrJ8)

- Best Blog: [Medium Blog](https://medium.com/@deepanshut041/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40)

Demo Code: []()

## Object Detection and Recognition Techniques

### 1. Viola-Jones Detector

- Best Video: [Viola-Jones Algorithm Explained](https://youtu.be/_QZLbR67fUU?si=iJ1kOpoCmloIUR5n)

- Best Blog: [Face Detection using Viola Jones Algorithm](https://towardsdatascience.com/viola-jones-algorithm-and-haar-cascade-classifier-ee3bfb19f7d8/)

- Demo Code: [Viola-Jones Face Detection in Python](https://github.com/parulnith/Face-Detection-in-Python-using-OpenCV)

### 2. R-CNN (Region-Based Convolutional Neural Networks)

- Best Video: [R-CNN](https://youtu.be/5DvljLV4S1E?si=--ykIMiWStX27IDA)
- Best Blog: [Mediumn]()
- Demo Code: [Github]()
### 3. Faster R-CNN

- Best Video: [Faster R-CNN, Towards Real-Time Object Detection](https://youtu.be/itjQT-gFQBY?si=ZtU7zQ0RbALlhcLn)

- Best Blog: [Understanding Faster R-CNN for Object Detection](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8)

- Demo Code: [Faster R-CNN Implementation in PyTorch](https://github.com/pytorch/vision/tree/main/torchvision/models/detection)

### 4. YOLO (You Only Look Once)

- Best Video: [YOLO v3 | Explanation & Implementation](https://www.youtube.com/watch?v=Grir6TZbc1M)

- Best Blog: [YOLO: Real-Time Object Detection Explained](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006)

- Demo Code: [YOLOv5 Implementation](https://github.com/ultralytics/yolov5)

### 5. SSD (Single Shot Multi-Box Detector)

- Best Video: [Single Shot Detector (SSD) Explained](https://www.youtube.com/watch?v=P8e-G-Mhx4k)

- Best Blog: [Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)

- Demo Code: [SSD: Single Shot MultiBox Detector implementation in Keras](https://github.com/pierluigiferrari/ssd_keras)

## Convolutional Neural Networks (CNNs) Pretrained with Models for Classification

### AlexNet

- Best Video: [AlexNet Explained | CNN Architecture](https://www.youtube.com/watch?v=FmpDIaiMIeA)
- Best Blog: [Understanding AlexNet](https://www.digitalocean.com/community/tutorials/popular-deep-learning-architectures-alexnet-vgg-googlenet)[3]

### VGGNet

- Best Video: [VGG16 - Convolutional Network / Architecture](https://www.youtube.com/watch?v=pSExXap-GNs)
- Best Blog: [VGG16 - Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/)

### ResNet (Residual Networks)

- Best Video: [ResNet (Deep Residual Learning) Explained](https://www.youtube.com/watch?v=GWt6Fu05voI)
- Best Blog: [Understanding ResNet](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)

### Inception Network (GoogLeNet)

- Best Video: [Inception Network Explained](https://www.youtube.com/watch?v=C86ZXvgpejM)
- Best Blog: [A Simple Guide to the Versions of the Inception Network](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)

## Transfer Learning

- Best Video: [Transfer Learning with TensorFlow Keras ](https://youtu.be/WWcgHjuKVqA?si=cQY_idQaNO6YVEIt)
- Another Video: [Transfer Learning with PyTorch](https://youtu.be/aPu6a5htRXM?si=re5LHQk4PiiNz_sG)
- Best Blog: [A Comprehensive Hands-on Guide to Transfer Learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)

## Image Segmentation

### Thresholding

- Best Video: [Image Thresholding - OpenCV with Python for Image and Video Analysis](https://www.youtube.com/watch?v=jXzkxsT9gxM)
- Best Blog: [Image Thresholding in OpenCV](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)

### K-means Clustering

- Best Video: [K Means Clustering Algorithm - Unsupervised Machine Learning](https://www.youtube.com/watch?v=4b5d3muPQmA)
- Best Blog: [K-Means Clustering in Python: A Practical Guide](https://realpython.com/k-means-clustering-python/)

### Fully Convolutional Networks (FCNs)

- Best Video: [Fully Convolutional Networks for Semantic Segmentation](https://www.youtube.com/watch?v=_aPVDQjXMNU)
- Best Blog: [Review: FCN — Fully Convolutional Network](https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1)

### U-Net

- Best Video: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://www.youtube.com/watch?v=azM57JuQpQI)
- Best Blog: [Understanding U-Net Architecture](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)

## Motion Analysis

### Optical Flow

- Best Video: [Optical Flow with OpenCV](https://www.youtube.com/watch?v=7A_yfQXPmXI)
- Best Blog: [Optical Flow in OpenCV](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)

### Kalman Filters

- Best Video: [Understanding Kalman Filters, Part 1: Why Use Kalman Filters?](https://www.youtube.com/watch?v=mwn8xhgNpFY)
- Best Blog: [Understanding Kalman Filters](https://www.mathworks.com/videos/series/understanding-kalman-filters.html)

## Advanced Techniques

### Generative Adversarial Networks (GANs)
- Best Intro Video: [GANS](https://youtu.be/TpMIssRdhco?si=Mf6MdhfZc1R9m92D)
- Best Theoritical Video: [GANS](https://youtu.be/RAa55G-oEuk?si=TA75Hrvt0VTA9axi)
- Best Video: [Generative Adversarial Networks (GANs) in 50 lines of code](https://www.youtube.com/watch?v=OljTVUVzPpM)
- Best Blog: [Understanding Generative Adversarial Networks (GANs)](https://medium.com/@marcodelpra/generative-adversarial-networks-dba10e1b4424)

### Self-Supervised Learning

- Best Video: [Self-Supervised Learning](https://www.youtube.com/watch?v=2KeUxXLJrZA)
- Best Blog: [Self-Supervised Representation Learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)

### Attention Mechanisms

- Best Video: [Attention Mechanism in Deep Learning](https://www.youtube.com/watch?v=W2rWgXJBZhU)
- Best Blog: [Attention in Neural Networks](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)


## Evaluation Metrics in Machine Learning

### Intersection Over Union (IOU)

- Best Video: [IOU](https://www.youtube.com/watch?v=duBGmrxNHS8)
- Best Blog: [Attention in Neural Networks](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)

### NMS 
- Best Video: [NMS](https://www.youtube.com/watch?v=duBGmrxNHS8)
- Best Blog: [NMS]()

### Mean Avarage Precision(map)

- Best Video: [MAP](https://www.youtube.com/watch?v=duBGmrxNHS8)
- Best Blog: [Attention in Neural Networks](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)


### Mean Intersection over Union (mIoU)
- Best Video: [Attention Mechanism in Deep Learning](https://www.youtube.com/watch?v=W2rWgXJBZhU)
- Best Blog: [Attention in Neural Networks](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)


### Pixcel Accuracy 


- Best Video: [Attention Mechanism in Deep Learning](https://www.youtube.com/watch?v=W2rWgXJBZhU)
- Best Blog: [Attention in Neural Networks](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)


### Dice Cofficient 

- Best Video: [Attention Mechanism in Deep Learning](https://www.youtube.com/watch?v=W2rWgXJBZhU)
- Best Blog: [Attention in Neural Networks](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)


