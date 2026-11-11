# Fitting Lines and Curves to Edges

Suppose we are given a set of edge pixels and want to fit a line, a low-order polynomial, or a curve to them.

## 1. Preprocessing an Edge Image

Consider an image of a vase. The desired boundary-detection output would contain clean, clear boundaries around the vase.

However, such an output may be highly aspirational: the example of clean boundaries may have been manually sketched. That is why the boundaries look so clean. Obtaining this result automatically is difficult.

> **Image placeholder: Vase boundary-detection goal**  
> Suggested visual: the original vase image and the manually sketched clean boundary that we would like to recover.

### Edge-Image Cleanup Pipeline

A practical preprocessing pipeline is:

1. **Apply edge detection.**  
   This produces an edge map. Each pixel contains the strength or magnitude of the edge at that location.

2. **Threshold the edge map.**  
   Thresholding converts the edge-strength image into a binary image.

3. **Use binary-image processing.**  
   One or more techniques from binary image processing can clean up the result.

4. **Shrink and expand the edges.**  
   Shrinking removes isolated edges. Expanding the remaining edges restores them, although they may now be thicker.

5. **Apply a thinning algorithm.**  
   Thinning reduces the thickness of the edges and produces a cleaner edge map.

The challenge is to move from this cleaned edge map to the actual object boundaries. That is the problem of boundary detection.

> **Image placeholder: Edge preprocessing pipeline**  
> Suggested visual: original image -> edge-strength map -> thresholded binary map -> shrink/expand cleanup -> thinned edge map -> detected boundary.

## 2. Fitting a Straight Line

Suppose we are given edge points $(x_i,y_i)$ and want to find the line

$$
y = mx + c
$$

that best represents them.

The unknown parameters are:

- $m$: the slope,
- $c$: the intercept.

### 2.1 Energy Based on Vertical Distance

First, define an energy or cost function. A reasonable first choice is the average squared **vertical distance** between each point and the estimated line.

For point $(x_i,y_i)$, the vertical residual is:

$$
y_i - mx_i - c
$$

The energy is therefore:

$$
E(m,c) = \frac{1}{N}\sum_{i=1}^{N}(y_i-mx_i-c)^2
$$

To fit the line, find $m$ and $c$ that minimize $E$.

### 2.2 Minimizing the Energy

Take the first derivative with respect to $m$ and set it to zero. Then take the first derivative with respect to $c$ and set it to zero:

$$
\frac{\partial E}{\partial m}=0,
\qquad
\frac{\partial E}{\partial c}=0
$$

The derivative can be taken inside the summation because differentiation and summation are both linear operators. Solving the resulting equations gives closed-form solutions:

$$
\bar{x}=\frac{1}{N}\sum_{i=1}^{N}x_i,
\qquad
\bar{y}=\frac{1}{N}\sum_{i=1}^{N}y_i
$$

$$
 m = \frac{\sum_{i=1}^{N}(x_i-\bar{x})(y_i-\bar{y})}
 {\sum_{i=1}^{N}(x_i-\bar{x})^2}
$$

$$
 c = \bar{y}-m\bar{x}
$$

Here, $\bar{x}$ and $\bar{y}$ are the means of the supplied edge-point coordinates. In practice, plug all $x_i$ and $y_i$ into the expressions to obtain $m$, then use $m$ to obtain $c$.

### 2.3 The Problem with Vertical Distance

The vertical-distance formulation can produce the wrong line.

For a set of points that visually lies along an oblique line, the desired best-fit line may be the orange line in the example. However, minimizing vertical distances can produce a line that is almost the opposite of the desired line, possibly even perpendicular to it.

Why does this happen? The energy function only measures vertical distance. For some points, the vertical distance to the desired line is large, even though their true geometric distance to the line is small. Another line can therefore achieve a smaller average squared vertical distance.

This shows that, when setting up a fitting problem, we must be careful about the energy function and, specifically, about which distance is being minimized.

> **Image placeholder: Vertical-distance fitting failure**  
> Suggested visual: the point set, the desired oblique line, vertical residual segments, and the incorrect line selected by vertical least squares.

## 3. Fitting Using Perpendicular Distance

A more reasonable choice is to minimize the average squared **perpendicular distance** from each point to the line.

For this, use the normal parameterization of a line:

$$
x\sin\theta - y\cos\theta + \rho = 0
$$

where:

- $\theta$ is the angle that the line makes with the horizontal axis,
- $\rho$ is the shortest distance from the line to the origin.

For a point $(x_i,y_i)$, the signed perpendicular-distance expression is:

$$
x_i\sin\theta-y_i\cos\theta+\rho
$$

The corresponding energy is:

$$
E(\theta,\rho)=\frac{1}{N}\sum_{i=1}^{N}
\left(x_i\sin\theta-y_i\cos\theta+\rho\right)^2
$$

This expression measures the squared perpendicular distance from every edge point to the candidate line.

> **Image placeholder: Perpendicular-distance line fitting**  
> Suggested visual: a line represented by $(\theta,\rho)$, the origin, and perpendicular distance segments from several edge points to the line.

### Connection to Binary-Image Moments

This energy is the same type of expression used when finding the axis of minimum second moment in a binary image. In that earlier setting, the expression was referred to as $r$.

Therefore, a set of edge points can be treated like a binary image:

1. Represent the edge points as a binary image.
2. Compute the three moments, traditionally denoted $a$, $b$, and $c$.
3. Use simple expressions involving $a$, $b$, and $c$ to obtain $\theta$ and $\rho$.

The resulting line is the line that minimizes the average squared perpendicular distance.

## 4. Fitting Curves with a Polynomial

Now suppose we are given edge points and want to fit a polynomial instead of a straight line.

As an example, use a third-order polynomial:

$$
y = ax^3 + bx^2 + cx + d
$$

The unknown parameters are $a$, $b$, $c$, and $d$.

### 4.1 Polynomial-Fitting Energy

Using the vertical-distance metric, define the energy as the average squared vertical distance between each point and the polynomial:

$$
E(a,b,c,d)=\frac{1}{N}\sum_{i=1}^{N}
\left[y_i-(ax_i^3+bx_i^2+cx_i+d)\right]^2
$$

The expression inside the square is the vertical distance between $(x_i,y_i)$ and the polynomial.

To minimize the energy, differentiate with respect to each unknown parameter and set the derivatives to zero:

$$
\frac{\partial E}{\partial a}=0,
\qquad
\frac{\partial E}{\partial b}=0,
\qquad
\frac{\partial E}{\partial c}=0,
\qquad
\frac{\partial E}{\partial d}=0
$$

Even though $x^3$ and $x^2$ appear in the expression, those values are known because the data points are given. The expression is linear in the unknowns $a$, $b$, $c$, and $d$. After differentiation, we therefore obtain a system of linear equations.

For a third-order polynomial, this gives four equations in four unknowns, which can be solved for $a$, $b$, $c$, and $d$.

### 4.2 Why a General Method Is Needed

Deriving the derivatives by hand works for a third-order polynomial. However, if we later want to fit a fifth-order polynomial or another order, we would have to derive a new set of equations each time.

Doing this case by case is cumbersome. We therefore want a general technique for solving the resulting system of linear equations.

## 5. Polynomial Fitting as a Linear System

Take the polynomial equation and substitute all the observed points:

$$
(x_0,y_0),\ (x_1,y_1),\ \ldots,\ (x_{n-1},y_{n-1})
$$

Each point produces one equation. For the third-order example, the unknowns are $a$, $b$, $c$, and $d$:

$$
\begin{aligned}
y_0 &= ax_0^3+bx_0^2+cx_0+d \\
y_1 &= ax_1^3+bx_1^2+cx_1+d \\
&\ \vdots \\
y_{n-1} &= ax_{n-1}^3+bx_{n-1}^2+cx_{n-1}+d
\end{aligned}
$$

The number of equations equals the number of supplied edge points. If the number of edge points is larger than the number of unknown coefficients, the system is **over-determined**.

> **Image placeholder: Polynomial samples and equations**  
> Suggested visual: edge points sampled from a curve, the fitted polynomial, and the corresponding equations generated by the samples.

## 6. Solving an Over-Determined Linear System

### 6.1 General Matrix Form

Suppose there are $m$ unknown coefficients and $n$ observations. In the lecture's notation:

- $i$ identifies the data point or edge observation,
- $j$ identifies the coefficient in the linear equation,
- $X_{ij}$ is the corresponding known matrix entry.

The system can be written as:

$$
X\mathbf{a}=\mathbf{y}
$$

where:

- $X$ is known and has dimensions $n\times m$,
- $\mathbf{y}$ is known and has dimensions $n\times 1$,
- $\mathbf{a}$ contains the unknown coefficients and has dimensions $m\times 1$.

For an over-determined system, $n>m$.

> **Image placeholder: Matrix representation**  
> Suggested visual: the matrix $X$, coefficient vector $\mathbf{a}$, and observation vector $\mathbf{y}$ with their dimensions.

### 6.2 Why the Ordinary Inverse Does Not Work

A first instinct might be to find $X^{-1}$ and solve the system. However, $X$ is not a square matrix in an over-determined system, so it is not directly invertible.

### 6.3 The Pseudoinverse Solution

Pre-multiply both sides of

$$
X\mathbf{a}=\mathbf{y}
$$

by $X^T$:

$$
X^TX\mathbf{a}=X^T\mathbf{y}
$$

The matrix $X^TX$ is square, with dimensions $m\times m$. Assuming it is invertible, solve for $\mathbf{a}$:

$$
\mathbf{a}=(X^TX)^{-1}X^T\mathbf{y}
$$

The matrix

$$
X^+=(X^TX)^{-1}X^T
$$

is called the **pseudoinverse** of $X$. Therefore, the solution can be written compactly as:

$$
\mathbf{a}=X^+\mathbf{y}
$$

This is a general technique for solving an over-determined system of linear equations.

## Key Takeaways

- Boundary detection converts edge pixels into clean object boundaries.
- Edge images can be cleaned using thresholding, shrink/expand operations, and thinning.
- A line can be fitted using $y=mx+c$, but minimizing vertical distance can give an undesirable result.
- Perpendicular-distance fitting uses the normal form $x\sin\theta-y\cos\theta+\rho=0$.
- The perpendicular-distance formulation connects to the axis of minimum second moment of a binary image.
- Polynomial fitting produces a linear system in the unknown coefficients.
- When there are more observations than unknowns, the system is over-determined.
- The pseudoinverse provides the general solution:

$$
\mathbf{a}=X^+\mathbf{y}
$$
