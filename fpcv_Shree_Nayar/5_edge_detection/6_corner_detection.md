# Corner Detection and the Harris Detector

A **corner** is a point where two or more edges meet. In a small image window around a corner, intensity changes significantly in more than one direction.

Corner detection uses image derivatives to distinguish flat regions, edges, and corners. The Harris corner detector summarizes the distribution of local gradients and assigns each location a corner-response score.

---

## 1. Flat Regions, Edges, and Corners

![Flat Region Edge and Corner](artifacts/flat-edge-corner.png)

Consider a small window placed at different locations in an image.

### Flat region

Intensity is approximately constant in every direction. The horizontal and vertical derivatives are both small, except for small fluctuations caused by noise.

### Edge region

Intensity changes strongly across the edge but changes little along it. The gradient responses are therefore strong mainly in one direction.

### Corner region

Two edges meet, so intensity changes strongly in two different directions. The window contains significant gradient responses in both directions.

This directional difference is the key to separating corners from ordinary edges.

---

## 2. Image Gradients in a Window

At each pixel in a window, compute the image derivatives:

$$
I_x=\frac{\partial I}{\partial x},
\qquad
I_y=\frac{\partial I}{\partial y}
$$

The pair $(I_x,I_y)$ describes the local intensity change. We can visualize the collection of gradient pairs in **gradient space**, where $I_x$ is one axis and $I_y$ is the other.

* In a flat region, most gradient pairs cluster near $(0,0)$.
* Along an edge, strong gradient pairs tend to form an elongated cluster in one direction.
* Around a corner, strong responses occur in two different directions, so the distribution spreads in both dimensions.

The goal is to summarize the shape and spread of this gradient distribution using a small number of values.

---

## 3. The Structure Tensor

The local gradient distribution can be summarized using the **structure tensor** (also called the second-moment matrix):

$$
M=
\sum_{(x,y)\in W}w(x,y)
\begin{bmatrix}
I_x^2 & I_xI_y\\
I_xI_y & I_y^2
\end{bmatrix}
=
\begin{bmatrix}
A & B\\
B & C
\end{bmatrix}
$$

where:

* $W$ is a neighborhood around the pixel being tested
* $w(x,y)$ is an optional weighting function, often a Gaussian centered on the neighborhood
* $A=\sum w I_x^2$
* $B=\sum w I_xI_y$
* $C=\sum w I_y^2$

The eigenvalues of $M$, denoted $\lambda_1$ and $\lambda_2$, describe how much the gradient distribution varies along two perpendicular directions. Geometrically, they describe the two principal axes of an ellipse fitted to the gradient distribution.

The eigenvalues help classify the local image structure:

| Eigenvalues | Interpretation |
|---|---|
| Both small | Flat region: little intensity change in any direction |
| One large, one small | Edge: strong change in one direction, weak change in the other |
| Both large | Corner: strong changes in two directions |

---

## 4. Computing the Eigenvalues

For:

$$
M=\begin{bmatrix}A&B\\B&C\end{bmatrix}
$$

the eigenvalues are:

$$
\lambda_{1,2}
=
\frac{A+C\pm\sqrt{(A-C)^2+4B^2}}{2}
$$

The larger eigenvalue corresponds to the direction of greatest variation; the smaller eigenvalue corresponds to the perpendicular direction.

Interpreting these values:

* **Flat region:** $\lambda_1$ and $\lambda_2$ are both small.
* **Edge:** one eigenvalue is large and the other is small.
* **Corner:** both eigenvalues are large.

Computing eigenvalues at every pixel is possible, but the Harris detector uses a convenient response formula that avoids an explicit eigenvalue calculation.

---

## 5. Harris Corner Response

![Harris Corner Response](artifacts/harris-response.png)

The Harris response is:

$$
R=\det(M)-k\,[\operatorname{trace}(M)]^2
$$

For the matrix above:

$$
\det(M)=AC-B^2
$$

and:

$$
\operatorname{trace}(M)=A+C
$$

Therefore:

$$
R=(AC-B^2)-k(A+C)^2
$$

The parameter $k$ is an empirical constant, commonly chosen as a small positive value. The response can also be expressed in terms of the eigenvalues:

$$
R=\lambda_1\lambda_2-k(\lambda_1+\lambda_2)^2
$$

Typical interpretations are:

* $R$ is small when both eigenvalues are small: likely a flat region.
* $R$ is negative when one eigenvalue dominates: likely an edge.
* $R$ is large and positive when both eigenvalues are large: likely a corner.

A threshold on $R$ selects candidate corners.

---

## 6. Detecting and Localizing Corners

![Harris Corner Detection Steps](artifacts/harris-corner-steps.png)

A high Harris response may appear across several neighboring pixels around a corner. The detector should report a single location rather than every pixel in that response cluster.

A typical process is:

1. Compute image gradients $I_x$ and $I_y$.
2. Form and sum the gradient products in a local window to create $M$.
3. Compute the Harris response $R$ at each location.
4. Threshold $R$ to keep corner candidates.
5. Apply **non-maximum suppression** to retain local peaks.

### Non-maximum suppression

Slide a small window over the response image. Keep a pixel only if its response is the maximum within that window; suppress neighboring responses that are lower.

This peak-detection step reduces clusters of nearby responses to representative corner locations. The remaining points can then be marked on the original image.

---

## 7. Example Applications

![Harris Corners on Text and Circuit Board](artifacts/harris-corner-examples.png)

On a simple image containing letters, strong Harris responses often occur at the junctions and sharp bends of the characters.

On a printed circuit board, many corners occur at component boundaries and trace junctions. Thresholding the response and applying non-maximum suppression can locate many of these points, although some true corners may be missed and some strong curved features may also be selected.

The choice of neighborhood size, response threshold, and suppression window affects the final detections.

---

## 8. A Corner-Related Visual Illusion

![Checkerboard Corner Illusion](artifacts/corner-bulge-illusion.png)

A checkerboard-like pattern can appear to bulge when small contrasting squares or dots are placed near its intersections. Without these small marks, the main grid lines are parallel or perpendicular. Adding the marks can bias the visual system to perceive the corner locations as shifted, making the center look curved or bulged.

This illusion illustrates that human perception of corners and line geometry can be influenced by nearby image structure. Human vision is powerful, but it is not infallible.

---

## 9. Summary

Corner detection distinguishes three types of local image regions by examining how gradients vary in a window:

* **Flat:** little variation in either direction.
* **Edge:** strong variation in one direction, weak variation in the other.
* **Corner:** strong variation in two directions.

The Harris detector:

1. computes image gradients
2. summarizes gradient variation with the structure tensor
3. calculates the response:
   $$
   R=\det(M)-k[\operatorname{trace}(M)]^2
   $$
4. thresholds the response
5. uses non-maximum suppression to keep localized peaks

The result is a set of candidate corner locations that can be used as features for later vision tasks.
