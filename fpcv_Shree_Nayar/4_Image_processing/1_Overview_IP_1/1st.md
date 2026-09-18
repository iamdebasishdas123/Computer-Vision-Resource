# Introduction to Image Processing

This is the first of two lectures devoted to image processing.

## What Is Image Processing?

In image processing, we are given an image and want to transform it into a clearer image or an image that is easier to analyze.

### Creating a Clearer Image

Image processing can be used to improve the clarity of an image in several situations:

- **Noise:** A scene captured at night may be grainy or noisy because of insufficient light. Image processing can remove this noise.
- **Motion blur:** A fast-moving object may be smeared in the captured image. Image processing can remove the smearing and create a crisp image.
- **Defocus blur:** An object may be out of focus during image capture. Image processing can remove the blur so that the object is crisp.

This is the business of creating an enhanced or clearer image.

> **Image placeholder: Image enhancement examples**  
> Suggested visual: a noisy image, a motion-blurred image, and a defocus-blurred image alongside their enhanced versions.

### Recovering Important Information

Image processing can also recover information from an image that is most salient to the computer-vision problem being solved. For example, we may want to enhance:

- edges in a scene,
- corners in a scene, or
- other interesting points in a scene.

Image processing tools lie under the hood in any computer-vision system and are extremely important.

## Topics in This Lecture

### 1. Pixel Processing

Pixel processing is the simplest type of image processing that can be applied.

For each pixel in the image:

1. Look at its brightness or color value.
2. Transform it using a predetermined mapping.

The operation is not concerned with where the pixel lies in the image. It is a transformation based only on the color or brightness value of that pixel.

### 2. Linear Shift-Invariant Systems

Linear shift-invariant systems are a very important class of systems in image processing. Many operations applied to images are linear and shift-invariant.

Any system that is linear and shift-invariant can be implemented as a convolution. We will therefore study:

- what convolution is,
- the properties of convolution, and
- how convolution follows from linear systems theory.

Based on convolution and linear systems theory, we can develop a suite of linear image filters. These filters are very simple to apply because they can be implemented using convolutions. We will see what kinds of modifications can be applied to an image using linear image filters.

### 3. Nonlinear Image Filters

There are certain things we want to do to an image that simply cannot be done using convolution. This takes us to the class of nonlinear image filters.

These filters can be viewed as more algorithmic in nature. For a pixel and its neighborhood, we look at the values in the neighborhood and apply a simple algorithm to produce the output value of that pixel.

> **Image placeholder: Neighborhood-based filtering**  
> Suggested visual: one target pixel, its surrounding neighborhood, and the resulting output pixel.

### 4. Template Matching

Template matching is the problem of finding a certain pattern everywhere it appears in an image.

This problem can be solved using **correlation**, which is related in some ways to the concept of convolution.
