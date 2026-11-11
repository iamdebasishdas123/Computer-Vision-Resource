# Overview of Boundary Detection

## From Edges to Object Boundaries

Once we know how to find edges in an image, the next goal is to turn those edge pixels into clean outlines or silhouettes of objects.

This is the problem of **boundary detection**:

$$
\text{edge pixels} \longrightarrow \text{object boundaries}
$$

> **Image placeholder: From edges to boundaries**  
> Suggested visual: an image, its edge map, and the final clean object outline or silhouette.

## Topics in Boundary Detection

### 1. Fitting Lines and Curves to Edges

Given a set of edges in an image, we can fit a low-order polynomial, line, or curve to those edges.

This problem can be set up as a system of linear equations. Such systems can be solved efficiently and robustly.

### 2. Active Contours (Snakes)

Active contours, also called **snakes**, are widely used in computer vision, especially in fields such as medical imaging.

Suppose an image contains an object whose boundary we want to find, along with many other things inside the image. The process is:

1. Sketch a very rough outline around the object, just outside it.
2. The initial outline can even be drawn by hand.
3. Iteratively modify or deform the contour.
4. Allow the contour to latch onto the actual object boundary like a rubber band.

We will study how to formulate the active-contour problem and how to compute active contours efficiently.

> **Image placeholder: Active contour evolution**  
> Suggested visual: an object, a rough initial contour outside it, and the contour deforming until it reaches the true boundary.

### 3. The Hough Transform

A difficult problem in boundary detection is determining which edges belong to the boundary we are looking for. This is an **inlier-outlier problem**:

- **Inliers:** edge pixels that belong to the desired boundary.
- **Outliers:** other edges in the image that do not belong to that boundary.

The **Hough Transform** is a clever technique, invented in the 1960s, that provides an elegant way to solve this inlier-outlier problem.

It can be set up to find simple geometric forms, such as:

- lines, and
- circles.

It can find these forms effectively even in complex images.

> **Image placeholder: Hough Transform for lines and circles**  
> Suggested visual: a complex edge image with a detected line and circle highlighted, plus the corresponding parameter-space peaks.

### 4. Generalized Hough Transform

The final topic is a generalization of the Hough Transform.

What if the object we are looking for is not given in a parametric form? What if it is a hand-drawn sketch or another complex shape?

The Hough Transform can be generalized to work with these kinds of shapes as well. We will study:

- how the generalized Hough Transform can be implemented, and
- the limitations of the generalized Hough Transform.
