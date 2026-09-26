# Image Processing: Second-Lecture Overview

In the first image-processing lecture, we studied methods for transforming an image into a clearer, enhanced, or more useful representation.

The main topics were pixel processing, linear shift-invariant systems, convolution, linear filters, nonlinear filters, and template matching.

---

## 1. Topics from the First Image-Processing Lecture

![Image Processing Overview](../artifacts/image-processing-overview.png)

### Pixel Processing

Pixel processing is the simplest type of image processing. It transforms pixel values individually, often to enhance contrast, change brightness, or modify the appearance of an image.

### Linear Shift-Invariant Systems

We then studied linear shift-invariant systems. These systems satisfy two properties:

* linearity
* shift invariance

We showed that every linear shift-invariant system performs a convolution, and that convolution itself is a linear shift-invariant operation.

### Linear Image Filters

Using this theory, we developed linear filters for tasks such as:

* smoothing images
* reducing noise
* enhancing useful image structures

### Nonlinear Image Filters

Some image-processing operations cannot be performed using convolution. We therefore studied nonlinear filters, including:

* the median filter
* the bilateral filter

These filters can reduce noise while preserving image structures more effectively in some situations.

### Template Matching

Finally, we studied template matching: the problem of finding a small pattern inside a larger image.

Correlation is an effective method for template matching and is closely related to convolution. The main difference is that correlation compares the template without flipping it.

---

## 2. Topics in the Second Image-Processing Lecture

In the second image-processing lecture, we begin working in the frequency domain.

The topics are:

1. Fourier transform
2. convolution in the frequency domain
3. deconvolution
4. sampling theory and aliasing

---

## 3. Fourier Transform

![Fourier Transform](../artifacts/fourier-transform-overview.png)

The Fourier transform allows us to represent an image or signal in the frequency domain instead of the spatial domain.

Many image-processing operations are easier to analyze and design in the frequency domain. Instead of describing an image only by its pixel values, we describe it using spatial frequencies and their amplitudes and phases.

---

## 4. Convolution in the Frequency Domain

![Convolution Theorem](../artifacts/convolution-theorem.png)

Convolution in the spatial domain is equivalent to multiplication in the Fourier domain.

If:

$$
g=f*h
$$

then:

$$
G(u,v)=F(u,v)H(u,v)
$$

where $F$, $G$, and $H$ are the Fourier transforms of $f$, $g$, and $h$.

This important result allows us to design and analyze many linear image filters in the frequency domain.

---

## 5. Deconvolution

![Image Deconvolution](../artifacts/deconvolution.png)

During image capture, an image may unintentionally be blurred by a process such as motion blur. The captured image can often be modeled as the convolution of the original image with a blur function.

If:

$$
g=f*h
$$

then, in the Fourier domain:

$$
G=FH
$$

Ideally, recovering the original image would require:

$$
F=\frac{G}{H}
$$

The process of attempting to undo the blur is called **deconvolution**.

In practice, noise and small values of $H$ make direct division unstable. Nevertheless, deconvolution is generally easier to formulate and solve in the frequency domain than in the spatial domain.

---

## 6. Sampling Theory

![Image Sampling](../artifacts/image-sampling.png)

An image sensor receives a continuous optical image. The sensor samples this image on a regular grid of pixels to create a digital image.

Sampling theory studies how the sampling frequency affects the information that can be recovered.

If the sampling frequency is sufficiently high, the continuous signal can be reconstructed without information loss under the assumptions of sampling theory.

If the signal is sampled too slowly, information is lost and unwanted patterns may appear.

---

## 7. Aliasing

![Aliasing](../artifacts/aliasing.png)

When an image is undersampled, high-frequency structures can appear as false lower-frequency patterns. This artifact is called **aliasing**.

Aliasing may appear as:

* false stripes
* repeating patterns
* jagged edges
* moiré patterns
* incorrect apparent motion

Cameras use optical or digital anti-aliasing techniques to reduce these artifacts before or during sampling.

---

## 8. Summary

The first image-processing lecture introduced spatial-domain methods:

* pixel processing
* convolution
* linear filters
* nonlinear filters
* template matching

The second lecture extends these ideas to the frequency domain. We study:

* Fourier representation
* the convolution theorem
* frequency-domain filter design
* deconvolution
* sampling
* aliasing

Together, these topics provide the mathematical foundation for analyzing, enhancing, restoring, and sampling digital images.
