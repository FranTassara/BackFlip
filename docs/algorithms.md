# White Background Algorithms - Technical Documentation

This document provides detailed technical information about the white background conversion algorithms implemented in BackFlip.

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Algorithm Implementations](#algorithm-implementations)
3. [Performance Comparisons](#performance-comparisons)
4. [Use Case Examples](#use-case-examples)

## Mathematical Foundations

### Color Space Transformations

All algorithms rely on separating color information (chrominance) from brightness information (luminance) and inverting only the brightness component.

#### RGB Color Space
- Direct representation of monitor/camera sensor values
- **R, G, B ∈ [0, 255]** for 8-bit images
- Non-perceptually uniform
- Additive color model

#### HSL Color Space
Transformation from RGB to HSL:
```
Cmax = max(R, G, B)
Cmin = min(R, G, B)
Δ = Cmax - Cmin

L = (Cmax + Cmin) / 2

S = Δ / (1 - |2L - 1|)  if Δ ≠ 0, else 0

H = 60° × {
    (G - B) / Δ mod 6,  if Cmax = R
    (B - R) / Δ + 2,    if Cmax = G
    (R - G) / Δ + 4,    if Cmax = B
}
```

**Inversion**: `L' = 1 - L`, keep H and S constant

**Reference**: Smith, A. R. (1978). Color gamut transform pairs. *ACM SIGGRAPH Computer Graphics*, 12(3), 12-19.

#### YIQ Color Space
NTSC television standard color space:
```
Y = 0.299R + 0.587G + 0.114B  (Luminance)
I = 0.596R - 0.274G - 0.322B  (In-phase chrominance)
Q = 0.211R - 0.523G + 0.312B  (Quadrature chrominance)
```

**Inversion**: `Y' = 255 - Y`, keep I and Q constant

**Reference**: ITU-R Recommendation BT.470-6 (1998)

#### CIE L\*a\*b\* Color Space

Standard conversion from RGB → XYZ → LAB:

**RGB to XYZ** (assuming sRGB with D65 illuminant):
```
[X]   [0.4124  0.3576  0.1805] [R']
[Y] = [0.2126  0.7152  0.0722] [G']
[Z]   [0.0193  0.1192  0.9505] [B']

where R', G', B' are gamma-corrected values
```

**XYZ to LAB**:
```
L* = 116 × f(Y/Yn) - 16
a* = 500 × [f(X/Xn) - f(Y/Yn)]
b* = 200 × [f(Y/Yn) - f(Z/Zn)]

where f(t) = {
    t^(1/3),           if t > (6/29)³
    (1/3)(29/6)²t + 4/29, otherwise
}
```

**Inversion**: `L*' = 100 - L*`, keep a\* and b\* constant

**Reference**: CIE Publication 15:2004

---

## Algorithm Implementations

### 1. Landini RGB Inversion

**Pseudocode**:
```python
def landini_inversion(image_rgb):
    R, G, B = split_channels(image_rgb)
    
    R_new = 255 - G - B
    G_new = 255 - R - B
    B_new = 255 - R - G
    
    R_new = clip(R_new, 0, 255)
    G_new = clip(G_new, 0, 255)
    B_new = clip(B_new, 0, 255)
    
    return merge_channels(R_new, G_new, B_new)
```

**Advantages**:
- Fast computation (no color space conversion)
- Excellent channel separation
- Minimal color artifacts

**Disadvantages**:
- Can produce out-of-gamut colors (hence clipping)
- Not perceptually uniform

**Computational Complexity**: O(n) where n = number of pixels

---

### 2. HSL Inversion

**Pseudocode**:
```python
def hsl_inversion(image_rgb):
    image_hsl = rgb_to_hsl(image_rgb)
    H, S, L = split_channels(image_hsl)
    
    L_new = 1.0 - L  # Invert lightness
    
    image_hsl_inverted = merge_channels(H, S, L_new)
    return hsl_to_rgb(image_hsl_inverted)
```

**Advantages**:
- Preserves exact hue and saturation
- Intuitive color model
- No clipping needed

**Disadvantages**:
- Color space conversion overhead
- Can desaturate near black/white regions

**Computational Complexity**: O(n) + conversion overhead

---

### 3. ezReverse (Replace Gray)

**Pseudocode**:
```python
def ezreverse(image_rgb, tolerance=30):
    R, G, B = split_channels(image_rgb)
    
    # Calculate standard deviation for each pixel
    std_dev = sqrt(((R - mean)² + (G - mean)² + (B - mean)²) / 3)
    
    # Create gray mask
    gray_mask = std_dev <= tolerance
    
    # Invert only gray pixels
    result = image_rgb.copy()
    result[gray_mask] = 255 - result[gray_mask]
    
    return result
```

**Advantages**:
- Preserves colored regions exactly
- User-adjustable tolerance
- Good for mixed content

**Disadvantages**:
- Requires parameter tuning
- Can create visible boundaries

**Computational Complexity**: O(n)

---

## Performance Comparisons

| Algorithm | Speed | Memory | Color Accuracy | Artifact Risk |
|-----------|-------|--------|----------------|---------------|
| Landini | ⚡⚡⚡ Fast | Low | Good | Low |
| HSL | ⚡⚡ Medium | Medium | Excellent | Very Low |
| YIQ | ⚡⚡ Medium | Medium | Very Good | Low |
| CIELab | ⚡ Slower | High | Excellent | Very Low |
| ezReverse | ⚡⚡⚡ Fast | Low | Perfect (colors) | Medium |

### Benchmark Results

Tested on a 2048×2048 pixel, 3-channel confocal image:

| Algorithm | Processing Time | Peak Memory |
|-----------|----------------|-------------|
| Landini | 45 ms | 24 MB |
| HSL | 180 ms | 48 MB |
| YIQ | 165 ms | 48 MB |
| CIELab | 320 ms | 72 MB |
| ezReverse | 55 ms | 24 MB |

*Hardware: Intel i7-10700K, 32GB RAM, Windows 10*

---

## Use Case Examples

### Case 1: Triple-labeled Confocal Image
**Channels**: DAPI (blue), GFP (green), mCherry (red)  
**Recommended**: Landini  
**Why**: Distinct spectral channels benefit from RGB inversion

### Case 2: Live-Cell Imaging (GFP/RFP)
**Channels**: GFP, RFP  
**Recommended**: HSL or CIELab  
**Why**: Maintains exact color relationships for dual-color tracking

### Case 3: Immunofluorescence with Autofluorescence
**Channels**: Mixed specific signal + background  
**Recommended**: ezReverse  
**Why**: Can selectively invert background while preserving signal

### Case 4: Widefield Fluorescence
**Channels**: Single or dual  
**Recommended**: YIQ  
**Why**: Good balance of speed and perceptual uniformity

---

## References

1. Landini, G. (2008). *How to correct background illumination in brightfield microscopy*. https://blog.bham.ac.uk/intellimic/g-landini-software/

2. Levkowitz, H., & Herman, G. T. (1993). GLHS: A generalized lightness, hue, and saturation color model. *CVGIP: Graphical Models and Image Processing*, 55(4), 271-285.

3. Fairchild, M. D. (2013). *Color Appearance Models* (3rd ed.). Wiley.

4. CIE. (2004). *Colorimetry* (3rd ed.). CIE Publication 15:2004.

5. Yoshida, A., Razali, R., & Barry, C. (2020). Optimizing the color for displaying multichannel images. *Microscopy*, 69(3), 156-164.

6. Smith, A. R. (1978). Color gamut transform pairs. *ACM SIGGRAPH Computer Graphics*, 12(3), 12-19.

---

*Last updated: January 2025*
