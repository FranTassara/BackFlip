# BackFlip

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17536854.svg)](https://doi.org/10.5281/zenodo.17536854)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A PyQt-based GUI tool for compositing multi-channel confocal microscopy images with publication-ready white or black backgrounds.

## Features

- Multi-channel support (CZI, TIFF, standard images)
- Z-stack projection (Maximum, Average, Sum)
- Multiple white background algorithms
  - Landini (RGB inversion)
  - HSL/YIQ/CIELab color space inversion
  - ezReverse gray replacement
- Per-channel controls:
  - LUT selection (Gray, Red, Green, Blue, Cyan, Magenta, Yellow, Custom RGB)
  - Contrast and brightness adjustment
  - Background removal filters (Gaussian, Top-hat, Median, Threshold)
- Background removal filters (Gaussian, Top-hat, Median, Threshold)
- Scale bar with customization
- Export to TIFF, PNG, JPEG

## Screenshots

![Main Interface](docs/images/screenshot_main.png)

### Before & After
| Black Background | White Background |
|------------------|------------------|
| ![Before](docs/images/example_black_bg.png) | ![After](docs/images/example_white_bg.png) |

## Installation

### Option 1: From Source
```bash
# Clone the repository
git clone https://github.com/FranTassara/BackFlip.git
cd BackFlip

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python BackFlip_GUI.py
```

### Option 2: Standalone Executable (Windows)

Download the latest executable from [Releases](https://github.com/FranTassara/BackFlip/releases/tag/v1.0.0).

**Windows**: `BackFlip_App_v1.0_Windows.exe` (128 MB)  

## Usage

1. **Load your image**: Click "Load Image" and select your CZI, TIFF, or standard image file
2. **Adjust channels**: Use the right panel to modify LUT, contrast, and apply filters for each channel
3. **Choose background**: Select white or black background with your preferred conversion method
4. **Add scale bar**: Enable and customize the scale bar in the left panel (calibration read from metadata)
5. **Export**: Click "Export Image" to save your publication-ready figure

## White Background Algorithms

BackFlip offers multiple algorithms for optimal results:

- **Landini (RGB)**: Gabriel Landini's channel inversion method - best for most multi-channel images
- **HSL/YIQ/CIELab Inversion**: Color space transformations that preserve hue while inverting lightness
- **ezReverse**: Gray pixel detection and replacement - ideal for images with pure grayscale backgrounds

Choose the method that works best for your specific image.

## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL v3)

## Acknowledgments

- White background algorithms inspired by [Gabriel Landini's ImageJ work](https://blog.bham.ac.uk/intellimic/g-landini-software/)
- Built with [PySide6](https://wiki.qt.io/Qt_for_Python) (Qt for Python)
- Thanks to the microscopy community for testing and feedback

**BackFlip** - *Flip your backgrounds, not your workflow.*
