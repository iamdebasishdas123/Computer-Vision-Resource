# The Fourier Transform

The Fourier transform is named after Joseph Fourier. Fourier lived during the time of Napoleon and worked on problems involving heat propagation through materials of different shapes.

His work led to one of the most important ideas in mathematics, science, and engineering: a signal can be represented as a weighted combination of sinusoids with different frequencies.

Fourier's original work was not immediately accepted as mathematically rigorous by mathematicians such as Lagrange and Euler. It took him several years to publish the result. Nevertheless, the Fourier transform has influenced virtually every area of science and engineering, and many technological advances would not have been possible without it.

---

## 1. Sinusoids

![Sinusoid](../artifacts/sinusoid.png)

A one-dimensional sinusoid can be written as:

$$
f(x)=A\sin(2\pi ux+\psi)
$$

where:

* $A$ is the amplitude
* $u$ is the frequency
* $1/u$ is the period when $u\neq0$
* $\psi$ is the phase shift

The amplitude controls the strength of the signal, the frequency controls how rapidly it oscillates, and the phase controls its horizontal shift.

---

## 2. Fourier Series

![Fourier Series](../artifacts/fourier-series.png)

Fourier's central result is that a periodic function can be represented, without loss of information, as a weighted sum of sinusoids.

For example, a square wave can be approximated by adding a small number of appropriately chosen sinusoids. As more terms are added, the approximation becomes more accurate. With infinitely many terms, the complete square wave is recovered under the usual mathematical conditions.

The sinusoids are not arbitrary. Their frequencies, amplitudes, and phases are determined by the Fourier transform or Fourier-series coefficients.

---

## 3. Spatial and Frequency Domains

![Spatial and Frequency Domains](../artifacts/spatial-frequency-domain.png)

Let $f(x)$ represent a signal in the spatial domain, where $x$ is the spatial coordinate.

The same signal can be represented in the frequency domain using the frequency variable $u$.

The frequency-domain representation records the contribution of each sinusoid:

* amplitude: how strongly the frequency is present
* phase: how the sinusoid is shifted

Both amplitude and phase are essential. Two sinusoids can have the same frequency and amplitude but still produce different signals if their phases differ.

---

## 4. Fourier Transform and Inverse Fourier Transform

The Fourier transform maps a spatial-domain signal to its frequency representation:

$$
f(x)\xrightarrow{\mathcal{F}}F(u)
$$

The inverse Fourier transform maps the frequency representation back to the original signal:

$$
F(u)\xrightarrow{\mathcal{F}^{-1}}f(x)
$$

Under the conditions required by the transform, this change of representation does not lose information.

For a one-dimensional signal, the Fourier transform is:

$$
F(u)=\int_{-\infty}^{+\infty}f(x)e^{-i2\pi ux}\,dx
$$

The inverse Fourier transform is:

$$
f(x)=\int_{-\infty}^{+\infty}F(u)e^{+i2\pi ux}\,du
$$

The transform uses the negative imaginary sign, while the inverse transform uses the positive imaginary sign.

---

## 5. Complex Exponentials and Euler's Formula

![Euler Formula](../artifacts/euler-formula.png)

The Fourier transform uses complex exponentials. Euler's formula states:

$$
e^{i\theta}=\cos\theta+i\sin\theta
$$

Here, $i=\sqrt{-1}$.

This identity shows that a complex exponential contains both cosine and sine components. It therefore provides a compact way to represent sinusoids with different amplitudes and phases.

Euler's formula can be derived by expanding the exponential, sine, and cosine functions using their Taylor series. The even-power terms form the cosine series, while the odd-power terms form the sine series.

---

## 6. Why the Fourier Transform Is Complex

The Fourier transform $F(u)$ is generally complex:

$$
F(u)=F_R(u)+iF_I(u)
$$

The real and imaginary parts together store the amplitude and phase of the sinusoid at frequency $u$.

The amplitude is:

$$
|F(u)|=\sqrt{F_R(u)^2+F_I(u)^2}
$$

The phase is:

$$
\phi(u)=\operatorname{atan2}\big(F_I(u),F_R(u)\big)
$$

The two-argument function $\operatorname{atan2}$ is preferred because it correctly determines the phase quadrant.

The Fourier transform uses both positive and negative frequencies. This is necessary for a complete representation of general signals.

---

## 7. Basic Fourier Transform Examples

![Fourier Transform Examples](../artifacts/fourier-transform-examples.png)

### 7.1 A Cosine

A cosine contains one frequency magnitude, represented at both $+k$ and $-k$:

$$
\cos(2\pi kx)\longleftrightarrow\frac{1}{2}\left[\delta(u-k)+\delta(u+k)\right]
$$

Thus, the Fourier transform contains impulses at $u=k$ and $u=-k$.

### 7.2 Two Cosines

For:

$$
f(x)=\cos(2\pi k_1x)+\cos(2\pi k_2x)
$$

the transform contains impulses at:

$$
u=\pm k_1,\qquad u=\pm k_2
$$

### 7.3 A Sine

A sine wave also contains the frequencies $+k$ and $-k$, but the phase relationship places its transform in the imaginary component.

### 7.4 A Constant Signal

A constant or flat signal contains only zero frequency:

$$
f(x)=C\longleftrightarrow C\delta(u)
$$

### 7.5 An Impulse

A spatial impulse contains all frequencies with equal strength. Therefore, its Fourier transform is constant across frequency.

### 7.6 A Rectangular Pulse

Consider a rectangular pulse of width $T$:

$$
f(x)=
\begin{cases}
1, & |x|\leq T/2\\
0, & \text{otherwise}
\end{cases}
$$

Its Fourier transform is:

$$
F(u)=T\,\operatorname{sinc}(Tu)
$$

where:

$$
\operatorname{sinc}(z)=\frac{\sin(\pi z)}{\pi z}
$$

The transform has a maximum at zero frequency and decaying oscillations, often called ringing.

### 7.7 A Gaussian

The Fourier transform of a Gaussian is also a Gaussian. The widths are inversely related: widening the Gaussian in the spatial domain narrows it in the frequency domain, and narrowing it in the spatial domain widens its frequency representation.

This inverse relationship between spatial width and frequency width is an important property of the Fourier transform.

---

## 8. Important Properties of the Fourier Transform

![Fourier Transform Properties](../artifacts/fourier-transform-properties.png)

### 8.1 Linearity

If:

$$
f_1(x)\longleftrightarrow F_1(u),\qquad f_2(x)\longleftrightarrow F_2(u)
$$

then:

$$
\alpha f_1(x)+\beta f_2(x)\longleftrightarrow\alpha F_1(u)+\beta F_2(u)
$$

### 8.2 Scaling

If $f(x)\longleftrightarrow F(u)$, then:

$$
f(ax)\longleftrightarrow\frac{1}{|a|}F\left(\frac{u}{a}\right)
$$

Stretching a signal in the spatial domain compresses its frequency representation, and compressing it in space widens its frequency representation.

### 8.3 Translation

If $f(x)\longleftrightarrow F(u)$, then shifting the signal by $a$ gives:

$$
f(x-a)\longleftrightarrow F(u)e^{-i2\pi ua}
$$

The magnitude is unchanged, but the phase changes according to the shift.

### 8.4 Differentiation

If $f(x)\longleftrightarrow F(u)$, then:

$$
\frac{d^nf(x)}{dx^n}\longleftrightarrow\left(i2\pi u\right)^nF(u)
$$

Differentiation in the spatial domain is therefore equivalent to multiplication by $(i2\pi u)^n$ in the frequency domain.

---

## 9. Summary

The Fourier transform represents a signal as a combination of sinusoids.

The main ideas are:

* a sinusoid is described by amplitude, frequency, and phase
* periodic signals can be represented by sums of sinusoids
* the Fourier transform maps spatial information to frequency information
* the inverse Fourier transform reconstructs the original signal
* Fourier coefficients are complex because they encode amplitude and phase
* positive and negative frequencies are required for a complete representation
* spatial width and frequency width have an inverse relationship
* linearity, scaling, translation, and differentiation provide useful transform properties

The Fourier transform is one of the central tools for analyzing and processing images in the frequency domain.
