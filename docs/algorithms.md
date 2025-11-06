# White Background Algorithms - Technical Documentation

This document provides detailed technical information about the white background conversion algorithms implemented in BackFlip.

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Algorithm Implementations](#algorithm-implementations)

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

**Reference**: Landini, G. (2008). How to correct background illumination in brightfield microscopy. ImageJ Documentation. Available at: https://blog.bham.ac.uk/intellimic/g-landini-software/

---

### 2. HSL Inversion

Converts to HSL (Hue, Saturation, Lightness) color space and inverts only the Lightness channel while preserving Hue and Saturation.

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

**Reference**:
- Song, X., & Goedhart, J. (2024). EzReverse – a Web Application for Background Adjustment of Color Images. bioRxiv. https://doi.org/10.1101/2024.05.27.594095
- The HSL color model is described in: Ramanath, R., & Drew, M. S. (2021). Color Spaces. In K. Ikeuchi (Ed.), Computer Vision: A Reference Guide (pp. 184–194). Springer International Publishing. https://doi.org/10.1007/978-3-030-63416-2_452

---

### 3. YIQ Inversion

Uses the NTSC YIQ color space, inverting luminance (Y) while preserving chrominance (I and Q).

Best for: Images requiring broadcast-standard color representation.

**Reference**:
- Song, X., & Goedhart, J. (2024). EzReverse – a Web Application for Background Adjustment of Color Images. bioRxiv. https://doi.org/10.1101/2024.05.27.594095
- The YIQ color model is described in: Ramanath, R., & Drew, M. S. (2021). Color Spaces. In K. Ikeuchi (Ed.), Computer Vision: A Reference Guide (pp. 184–194). Springer International Publishing. https://doi.org/10.1007/978-3-030-63416-2_452

---

### 4. CIELab Inversion

Converts to the CIE L*a*b* perceptually uniform color space and inverts the L* (lightness) channel.

Best for: Images requiring device-independent color representation and perceptually accurate lightness inversion.

**Reference**:
- Song, X., & Goedhart, J. (2024). EzReverse – a Web Application for Background Adjustment of Color Images. bioRxiv. https://doi.org/10.1101/2024.05.27.594095
- The CIELab color space is described in: Ramanath, R., & Drew, M. S. (2021). Color Spaces. In K. Ikeuchi (Ed.), Computer Vision: A Reference Guide (pp. 184–194). Springer International Publishing. https://doi.org/10.1007/978-3-030-63416-2_452

---

### 5. Replace Gray (ezReverse Method)

Detects near-grayscale pixels (where R≈G≈B) using standard deviation thresholding and inverts only those pixels, leaving colored pixels unchanged.

Algorithm: Pixels with std(R,G,B) < threshold are inverted, others remain unchanged.

Best for: Images with predominantly grayscale backgrounds and clear distinction between colored and gray regions. This method maintains hues identical to the original.

**Reference**:
- Song, X., & Goedhart, J. (2024). EzReverse – a Web Application for Background Adjustment of Color Images. bioRxiv. https://doi.org/10.1101/2024.05.27.594095

---

## References

1. **Song, X., & Goedhart, J. (2024).** EzReverse – a Web Application for Background Adjustment of Color Images. *bioRxiv*. https://doi.org/10.1101/2024.05.27.594095
   - *Primary source for HSL, YIQ, CIELab, and Replace Gray methods*

2. **Landini, G. (2008).** How to correct background illumination in brightfield microscopy. ImageJ Documentation. https://blog.bham.ac.uk/intellimic/g-landini-software/
   - *Source for RGB channel inversion method*

3. **Ramanath, R., & Drew, M. S. (2021).** Color Spaces. In K. Ikeuchi (Ed.), *Computer Vision: A Reference Guide* (pp. 184–194). Springer International Publishing. https://doi.org/10.1007/978-3-030-63416-2_452
   - *Foundational reference for color space theory*

4. **Johnson, J. (2012).** Not seeing is not believing: improving the visibility of your fluorescence images. *Molecular Biology of the Cell*, 23(5), 754–757. https://doi.org/10.1091/mbc.e11-09-0824
   - *Benefits of inverted backgrounds in fluorescence imaging*

```

*Last updated: January 2025*
