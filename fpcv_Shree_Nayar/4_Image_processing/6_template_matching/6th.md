# Template Matching Using Correlation

Template matching is the problem of finding a small image, or **template**, inside a larger image.

The template may represent an object, a pattern, or a small image region. The goal is to determine where that same pattern occurs in the larger image.

---

## 1. The Template-Matching Problem

![Template Matching](../artifacts/template-matching.png)

Suppose we are given:

* a large image $f$
* a smaller template $t$

We want to find the location where the template appears in the image.

The basic approach is to slide the template over the image. At every location, we compare the template with the image region underneath it.

If the two regions are very similar, we declare that the template has been found.

---

## 2. Sum of Squared Differences

![Sum of Squared Differences](../artifacts/template-squared-difference.png)

One way to measure the difference between the template and an overlapping image region is the **sum of squared differences**.

For a template positioned at $(i,j)$, define:

$$
E[i,j]=\sum_m\sum_n\left(f[i+m,j+n]-t[m,n]\right)^2
$$

The procedure is:

1. Subtract each template value from the corresponding image value.
2. Square every difference.
3. Add all squared differences.

The best match is the location where $E[i,j]$ is smallest. If $E[i,j]=0$, the overlapping image region is exactly equal to the template.

---

## 3. From Squared Difference to Correlation

Expand the squared term:

$$
(f-t)^2=f^2+t^2-2ft
$$

Therefore:

$$
E[i,j]=\sum f^2+\sum t^2-2\sum ft
$$

For a fixed template, the term $\sum t^2$ is constant. If the image energy is also fixed or changes only slightly, minimizing $E[i,j]$ is equivalent to maximizing:

$$
\sum_m\sum_n f[i+m,j+n]t[m,n]
$$

This product-and-sum operation is called **cross-correlation**.

---

## 4. Cross-Correlation

The cross-correlation between an image $f$ and a template $t$ is:

$$
R_{tf}[i,j]=\sum_m\sum_n t[m,n]f[i+m,j+n]
$$

The location with the largest correlation value is considered the best match.

---

## 5. Correlation Compared with Convolution

![Correlation and Convolution](../artifacts/correlation-convolution.png)

For discrete images, convolution is:

$$
(f*h)[i,j]=\sum_m\sum_n f[m,n]h[i-m,j-n]
$$

Cross-correlation is:

$$
R_{tf}[i,j]=\sum_m\sum_n t[m,n]f[i+m,j+n]
$$

The important difference is that convolution flips the filter, while correlation does not.

> Correlation is similar to convolution, but without flipping the template.

This distinction is important in template matching because we want to compare the template exactly as it is given.

---

## 6. Why Unnormalized Correlation Can Fail

![Unnormalized Correlation](../artifacts/unnormalized-correlation.png)

Consider a one-dimensional example with three possible locations: $A$, $B$, and $C$.

Suppose the template matches the signal exactly at $A$. However, unnormalized correlation can produce a larger value at $B$ or $C$ if the signal values there have larger magnitudes. The correlation depends not only on the pattern but also on the brightness and energy of the image region.

Consequently, the ordering may be:

$$
R_{tf}[C] > R_{tf}[B] > R_{tf}[A]
$$

This is a problem because $A$ is the true match.

---

## 7. Normalized Cross-Correlation

![Normalized Cross-Correlation](../artifacts/normalized-cross-correlation.png)

Normalize the correlation by the energy of the image region and the energy of the template:

$$
\rho[i,j]=\frac{\sum_m\sum_n t[m,n]f[i+m,j+n]}{\sqrt{\left(\sum_m\sum_n t[m,n]^2\right)\left(\sum_m\sum_n f[i+m,j+n]^2\right)}}
$$

The numerator measures similarity. The denominator reduces the effect of the magnitudes of the two signals.

The normalized correlation is usually bounded by:

$$
-1\leq\rho[i,j]\leq1
$$

For an exact match with the same sign and brightness pattern, $\rho[i,j]=1$. The best match is the location where $\rho[i,j]$ is largest.

---

## 8. Why Normalization Helps

Normalization makes template matching less sensitive to changes in brightness and camera gain.

The template may have been captured under one lighting condition, while the same pattern in the larger image may be brighter or darker because of:

* different illumination
* camera exposure
* sensor gain
* changes in scene brightness

The numerator measures pattern similarity, while the denominator compensates for signal energy. The correct location can therefore still produce the strongest response.

---

## 9. Example of Normalized Matching

![Normalized Matching Example](../artifacts/normalized-matching-example.png)

For candidate locations $A$, $B$, and $C$, normalized correlation gives the desired ordering:

$$
\rho[A]>\rho[B]>\rho[C]
$$

The response is determined primarily by how closely the shapes match, rather than by which region has the largest intensity values.

---

## 10. Template Matching in a Real Image

![Template Matching Result](../artifacts/template-matching-result.png)

Evaluating the template at every possible location produces a **correlation map** or **response map**. Each point records the similarity between the template and the corresponding image region.

The brightest point in a normalized cross-correlation map represents the strongest match.

The complete procedure is:

1. Choose a template.
2. Slide it over the image.
3. Compute normalized correlation at every valid location.
4. Find the maximum response.
5. Report that location as the template position.

---

## 11. Summary

Template matching finds a small pattern inside a larger image.

The main ideas are:

* sum of squared differences finds the smallest error
* expanding the squared error reveals a product-and-sum similarity term
* cross-correlation computes this product-and-sum operation
* correlation differs from convolution because it does not flip the template
* unnormalized correlation can be dominated by brightness or signal energy
* normalized cross-correlation reduces this problem
* the maximum response in the correlation map indicates the template location

Template matching is useful when the object has nearly the same appearance, orientation, and scale as the template.
