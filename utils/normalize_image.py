import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pickle import load
from skimage import exposure

# Histogram clipping + scaling parameters
MIN_VALUES = [2, 2, 1, 5]
MAX_VALUES = [25050, 27939, 28000, 27945]
P1 = [734, 581, 335, 364]
P99 = [7381, 6806, 7352, 6901]

# Load MinMaxScaler
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
scaler = load(open(SCALER_PATH, 'rb'))

def normalize_sentinel_image(input_path, output_path=None):
    """
    Normalize RGB+NIR Sentinel-2 image using histogram clipping and MinMaxScaler.
    
    Args:
        input_path (str): Path to the input .tif file.
        output_path (str, optional): If given, saves the normalized image to this path.
    
    Returns:
        np.ndarray: Normalized image array with shape (H, W, 4)
    """
    with rasterio.open(input_path) as src:
        bands = src.read()  # shape: (bands, height, width)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    # Rearrange to (H, W, C)
    bands = np.moveaxis(bands, 0, -1)

    # Extract bands: B02 (1), B03 (2), B04 (3), B08 (7)
    try:
        selected = bands[:, :, [1, 2, 3, 7]]
    except IndexError:
        raise ValueError("Input image must contain at least 8 bands.")

    # Histogram clipping
    clipped = np.zeros_like(selected, dtype=np.float32)
    for i in range(4):
        clipped[:, :, i] = exposure.rescale_intensity(
            selected[:, :, i],
            in_range=(P1[i], P99[i]),
            out_range=(MIN_VALUES[i], MAX_VALUES[i])
        )

    # MinMax scaling
    flat = clipped.reshape(-1, 4)
    scaled = scaler.transform(flat).reshape(clipped.shape).astype(np.float32)

    # Save if path is given
    if output_path:
        profile.update({
            'count': 4,
            'dtype': 'float32',
            'height': scaled.shape[0],
            'width': scaled.shape[1],
            'transform': transform,
            'crs': crs
        })
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(np.moveaxis(scaled, -1, 0))
        print(f"[INFO] Normalized image saved to: {output_path}")

    return scaled
