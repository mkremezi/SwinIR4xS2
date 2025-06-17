import os
import glob
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from affine import Affine
import fiona
from skimage.transform import resize
from shapely.geometry import shape

# Expected band resolutions
BAND_RESOLUTION = {
    'B01': '60m', 'B02': '10m', 'B03': '10m', 'B04': '10m',
    'B05': '20m', 'B06': '20m', 'B07': '20m', 'B08': '10m',
    'B8A': '20m', 'B09': '60m', 'B10': '60m', 'B11': '20m', 'B12': '20m'
}

DESIRED_BAND_ORDER = [
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
    'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
]

def get_band_files(safe_folder):
    pattern = os.path.join(safe_folder, "GRANULE", "*", "IMG_DATA", "*.jp2")
    files = glob.glob(pattern)
    bands = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            band_id = fname.split('_')[-1].split('.')[0]
            bands[band_id] = fpath
        except Exception as e:
            print(f"[WARNING] Could not parse band ID from {fname}: {e}")
    return bands

def read_band(band_path, target_shape=None):
    with rasterio.open(band_path) as src:
        band = src.read(1)
        profile = src.profile
    if target_shape and band.shape != target_shape:
        band = resize(band, target_shape, order=1, preserve_range=True, mode='constant').astype(profile['dtype'])
    return band, profile

def crop_image_with_shp_from_array(image_array, profile, shp_file):
    with fiona.open(shp_file) as shp:
        shp_crs = shp.crs
        geometries = []
        for feature in shp:
            geom = shape(feature['geometry'])
            if not geom.is_valid:
                print("[WARNING] Invalid geometry skipped.")
                continue
            try:
                transformed = transform_geom(shp_crs, profile['crs'], feature['geometry'])
                geometries.append(transformed)
            except Exception as e:
                print(f"[ERROR] Geometry transformation failed: {e}")

    mem_profile = profile.copy()
    mem_profile.update({'count': image_array.shape[-1]})
    image_for_mask = np.moveaxis(image_array, -1, 0)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**mem_profile) as dataset:
            dataset.write(image_for_mask)
            cropped_array, cropped_transform = mask(dataset, geometries, crop=True)

    cropped_image = np.moveaxis(cropped_array, 0, -1)
    return cropped_image, cropped_transform

def resample_and_crop_sentinel2(safe_folder, shp_file, output_file):
    band_files = get_band_files(safe_folder)
    if not band_files:
        raise ValueError("No JP2 bands found.")

    # Get target shape from 10m reference band
    target_shape = None
    for band_id in ['B02', 'B03', 'B04', 'B08']:
        if band_id in band_files:
            ref_array, ref_profile = read_band(band_files[band_id])
            target_shape = ref_array.shape
            print(f"[INFO] Using {band_id} as 10m reference: {target_shape}")
            break
    if target_shape is None:
        raise ValueError("No 10m reference band found.")

    band_arrays = []
    used_profile = None
    for band_id in DESIRED_BAND_ORDER:
        if band_id not in band_files:
            print(f"[WARNING] Band {band_id} missing. Skipping.")
            continue
        band_array, profile = read_band(band_files[band_id], target_shape=target_shape)
        band_arrays.append(band_array)
        used_profile = profile  # keep latest valid

    if not band_arrays:
        raise ValueError("No bands loaded.")
    image_stack = np.stack(band_arrays, axis=-1)
    print(f"[INFO] Stacked image shape: {image_stack.shape}")

    # Adjust profile transform for 10m output
    original_transform = used_profile['transform']
    pixel_size_ratio = original_transform.a / 10.0
    new_transform = Affine(
        original_transform.a / pixel_size_ratio,
        original_transform.b,
        original_transform.c,
        original_transform.d,
        original_transform.e / pixel_size_ratio,
        original_transform.f
    )

    used_profile.update({
        'transform': new_transform,
        'width': target_shape[1],
        'height': target_shape[0]
    })

    cropped_image, cropped_transform = crop_image_with_shp_from_array(image_stack, used_profile, shp_file)
    print(f"[INFO] Cropped image shape: {cropped_image.shape}")

    out_profile = used_profile.copy()
    out_profile.update({
        'driver': 'GTiff',
        'height': cropped_image.shape[0],
        'width': cropped_image.shape[1],
        'transform': cropped_transform,
        'count': cropped_image.shape[2],
        'dtype': cropped_image.dtype
    })

    with rasterio.open(output_file, "w", **out_profile) as dst:
        dst.write(np.moveaxis(cropped_image, -1, 0))
    print(f"[DONE] Output saved to: {output_file}")
