import numpy as np
import os
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from pickle import load
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler

# Load scaler from the same directory as this script
import pathlib
scaler_path = pathlib.Path(__file__).parent / "scaler.pkl"
scaler = load(open(scaler_path, 'rb'))

# Histogram clipping values (from training)
min_vals = [2, 2, 1, 5]
max_vals = [25050, 27939, 28000, 27945]
p1_vals = [734, 581, 335, 364]
p99_vals = [7381, 6806, 7352, 6901]

def reverse_scaling(
    super_res_image: np.ndarray,
    geo_transform=None,
    crs=None,
    save_path=None
) -> np.ndarray:
    """
    Reverse normalization and clipping of a 4-band super-resolved image.

    Parameters:
    - super_res_image: np.ndarray of shape (H, W, 4)
    - geo_transform: affine.Affine, optional
    - crs: rasterio CRS or dict, optional
    - save_path: str or None, if provided, saves GeoTIFF to this path

    Returns:
    - restored_image: np.ndarray of shape (H, W, 4), dtype=np.uint16
    """
    rows, cols, bands = super_res_image.shape
    assert bands == 4, "Expected 4 bands"

    # Reshape for inverse transform
    flat = super_res_image.reshape(-1, bands)
    restored = scaler.inverse_transform(flat)
    restored = restored.reshape(rows, cols, bands).astype(np.float32)

    # Reverse histogram clipping
    unclipped = np.zeros_like(restored, dtype=np.float32)
    for i in range(bands):
        unclipped[:, :, i] = exposure.rescale_intensity(
            restored[:, :, i],
            in_range=(min_vals[i], max_vals[i]),
            out_range=(p1_vals[i], p99_vals[i])
        )

    final_image = unclipped.astype(np.uint16)

    if save_path:
        # Update the geo_transform to reflect the new pixel size
        scaling_factor=4
        new_transform = geo_transform * geo_transform.scale(1 / scaling_factor, 1 / scaling_factor)

        with rasterio.open(
            save_path,
            "w",
            driver="GTiff",
            height=final_image.shape[0],
            width=final_image.shape[1],
            count=bands,
            dtype='uint16',
            crs=crs,
            transform=new_transform,
        ) as dst:
            for i in range(bands):
                dst.write(final_image[:, :, i], i + 1)

    return final_image
