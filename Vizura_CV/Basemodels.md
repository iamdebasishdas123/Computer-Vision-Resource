# CNN Architecture Comparison

## Comparison table

| Model | Year | Main architectural idea | Typical depth | Relative computation and parameters | Main strengths | Main limitations |
|---|---:|---|---:|---|---|---|
| **AlexNet** | 2012 | Stacked convolutions followed by fully connected layers | 8 layers | High for its era; ~60M parameters | Started the modern deep-learning era; strong ImageNet results | Large fully connected layers; obsolete and inefficient by current standards |
| **VGGNet** | 2014 | Repeated small `3×3` convolutions with pooling | 16–19 layers | Very high; VGG-16 has ~138M parameters | Simple, uniform design; useful feature extractor | Slow, memory-intensive, and over-parameterized |
| **Inception v1 (GoogLeNet)** | 2014 | Parallel `1×1`, `3×3`, `5×5` convolutions and pooling in an Inception module | 22 layers | Much lower than VGG; ~6.8M parameters | Multi-scale features with good accuracy and efficiency | More complex to design and implement; difficult to modify manually |
| **ResNet** | 2015 | Residual/skip connections: learn a residual instead of a full mapping | 18–152+ layers | Moderate; varies by version | Enables very deep networks; stable optimization; strong accuracy | Extra shortcut memory and compute; plain residual blocks can be costly at scale |
| **SqueezeNet** | 2016 | Fire modules: `1×1` squeeze layers and `1×1`/`3×3` expand layers | 18 layers | Very low; ~1.25M parameters | Small model size; suitable for storage-limited devices | Accuracy and latency may be worse than newer efficient models |
| **MobileNet** | 2017 | Depthwise-separable convolutions | 28 layers (V1 example) | Low; width and resolution can be scaled | Fast and compact for mobile/edge devices | Lower accuracy than larger models; hardware support affects speed |
| **DenseNet** | 2017 | Each layer receives features from all preceding layers by concatenation | 121–264 layers | Moderate parameters, but high feature-memory use | Excellent feature reuse and gradient flow; strong accuracy | Concatenation increases memory traffic and implementation complexity |
| **EfficientNet** | 2019 | Compound scaling of depth, width, and input resolution | Varies by B0–B7 | Excellent accuracy/efficiency trade-off | High accuracy with relatively few parameters; principled scaling | Training and deployment can be more complex; less flexible for some custom tasks |

## Architectures, innovations, and trade-offs

### 1. AlexNet

- **Architecture:** Five convolutional layers, three fully connected layers, ReLU activations, max pooling, and dropout.
- **Key innovation:** Demonstrated that deep CNNs trained on GPUs with ReLU and data augmentation could outperform traditional computer-vision methods.
- **Pros:** Simple to understand; historically important; faster training than sigmoid-based networks.
- **Cons:** Very large dense layers, high parameter count, and comparatively weak accuracy and efficiency today.

### 2. VGGNet

- **Architecture:** Uses a uniform sequence of `3×3` convolutions, with pooling between blocks, followed by fully connected layers.
- **Key innovation:** Showed that increasing depth with small filters can improve visual representation quality.
- **Pros:** Predictable structure; easy to implement; strong transfer-learning features.
- **Cons:** Extremely large model size, high memory use, and slow inference.

### 3. Inception v1 (GoogLeNet)

- **Architecture:** An Inception module processes the same feature map through parallel `1×1`, `3×3`, and approximate `5×5` paths plus pooling, then concatenates the results. `1×1` convolutions reduce channels before expensive operations.
- **Key innovation:** Captured features at multiple spatial scales while controlling computation.
- **Pros:** Much more parameter-efficient than VGG; combines local and broad visual patterns.
- **Cons:** Branches make the architecture harder to build, tune, and optimize on some hardware.

### 4. ResNet

- **Architecture:** Residual blocks add the block input directly to its output: `y = F(x) + x`. Projection shortcuts handle changed channel counts or spatial sizes.
- **Key innovation:** Skip connections reduce vanishing-gradient and degradation problems in very deep networks.
- **Pros:** Easy to optimize at great depth; reliable baseline; widely supported and adaptable.
- **Cons:** Larger variants require substantial compute; residual additions do not eliminate all optimization or memory costs.

### 5. SqueezeNet

- **Architecture:** Replaces most `3×3` filters with cheaper `1×1` filters. A Fire module contains a squeeze layer followed by parallel expand layers.
- **Key innovation:** Achieved AlexNet-level accuracy with a very small parameter footprint.
- **Pros:** Compact model files; lower storage and bandwidth requirements; useful for embedded systems.
- **Cons:** Small size does not always mean lowest latency; accuracy is behind many modern mobile architectures.

### 6. MobileNet

- **Architecture:** A depthwise convolution filters each input channel separately, followed by a pointwise `1×1` convolution that mixes channels.
- **Key innovation:** Depthwise-separable convolution greatly reduces computation compared with standard convolution.
- **Pros:** Efficient inference; configurable width and input-resolution multipliers; well suited to phones and edge devices.
- **Cons:** Accuracy-efficiency trade-offs can be significant at very small widths; actual speed depends on accelerator support.

### 7. DenseNet

- **Architecture:** Within a dense block, each layer receives the concatenated outputs of all earlier layers. Transition layers use `1×1` convolution and pooling.
- **Key innovation:** Feature reuse and direct connections improve gradient propagation without relearning the same features.
- **Pros:** Parameter-efficient; strong gradient flow; often performs well with fewer parameters.
- **Cons:** Feature concatenation consumes memory and bandwidth; dense connectivity complicates deployment.

### 8. EfficientNet

- **Architecture:** Builds on efficient convolutional blocks and scales network depth, width, and input resolution together using a compound scaling rule.
- **Key innovation:** Balanced scaling rather than increasing only one model dimension.
- **Pros:** Strong accuracy per parameter and per FLOP; multiple model sizes support different hardware budgets.
- **Cons:** Some versions use specialized blocks that are less convenient to customize; larger variants still require significant resources.

## Quick selection guide

- **For learning CNN history:** AlexNet or VGGNet.
- **For a strong general-purpose baseline:** ResNet.
- **For a compact model file:** SqueezeNet.
- **For mobile or edge inference:** MobileNet.
- **For feature reuse and transfer learning:** DenseNet.
- **For a strong accuracy-efficiency balance:** EfficientNet.
- **For multi-scale feature extraction:** Inception v1.