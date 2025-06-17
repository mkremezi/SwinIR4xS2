import os
import numpy as np
import torch
import math
import rasterio
from models.network_swinir import SwinIR

def swinir_inference(
    image_array,
    geo_transform=None,
    crs=None,
    save_path=None,
    device="cuda",
    scaling_factor=4,
    patch_size=64,
    border_thickness=20
):
    """
    Run SwinIR inference on a 4-band normalized image array.

    Parameters:
    - image_array: np.ndarray, shape (H, W, 4)
    - geo_transform: Affine or None, for writing output GeoTIFF
    - crs: str or dict or None, for writing output GeoTIFF
    - save_path: str or None, if provided, output will be saved to this path
    - device: "cuda" or "cpu"

    Returns:
    - super_res_image: np.ndarray, shape (H*scale, W*scale, 4)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    bands = image_array.shape[2]
    rows, cols = image_array.shape[:2]

    # Load model weights from models/swinir_weights.pth (relative to this file)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_weights_path = os.path.join(script_dir, "models", "swinir_weights.pth")

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"SwinIR weights not found at: {model_weights_path}")

    model = SwinIR(
        upscale=scaling_factor, img_size=(patch_size, patch_size), in_chans=4,
        window_size=8, img_range=1., depths=[3, 3, 3],
        embed_dim=60, num_heads=[3, 3, 3], mlp_ratio=2,
        upsampler='nearest+conv', resi_connection="1conv"
    ).to(device)

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # Padding
    impad_rows = math.ceil(rows / patch_size) * patch_size + 2 * patch_size
    impad_cols = math.ceil(cols / patch_size) * patch_size + 2 * patch_size
    padded = np.zeros((bands, impad_rows, impad_cols), dtype=np.float32)
    padded[:, patch_size:rows + patch_size, patch_size:cols + patch_size] = image_array.transpose(2, 0, 1)

    output_accumulator = []
    for start in [0, 15, 50, 102, 155, 205]:
        num_patches_x = (impad_cols - start) // patch_size
        num_patches_y = (impad_rows - start) // patch_size

        sub_img = padded[:, start:num_patches_y * patch_size + start, start:num_patches_x * patch_size + start]
        patches = sub_img.reshape(bands, num_patches_y, patch_size, num_patches_x, patch_size)
        patches = patches.transpose(1, 3, 0, 2, 4).reshape(-1, bands, patch_size, patch_size)

        predictions = np.zeros((patches.shape[0], bands, patch_size * scaling_factor, patch_size * scaling_factor), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, patches.shape[0], 32):
                input_tensor = torch.tensor(patches[i:i + 32]).to(device)
                pred = model(input_tensor).cpu().numpy()
                pred[:, :, :border_thickness, :] = 0
                pred[:, :, -border_thickness:, :] = 0
                pred[:, :, :, :border_thickness] = 0
                pred[:, :, :, -border_thickness:] = 0
                predictions[i:i + 32] = pred

        preds_reshaped = predictions.reshape(num_patches_y, num_patches_x, bands, patch_size * scaling_factor, patch_size * scaling_factor)
        preds_transposed = preds_reshaped.transpose(2, 0, 3, 1, 4).reshape(bands, num_patches_y * patch_size * scaling_factor, num_patches_x * patch_size * scaling_factor)

        full_out = np.zeros((bands, impad_rows * scaling_factor, impad_cols * scaling_factor), dtype=np.float32)
        full_out[:, start * scaling_factor:num_patches_y * patch_size * scaling_factor + start * scaling_factor,
                 start * scaling_factor:num_patches_x * patch_size * scaling_factor + start * scaling_factor] = preds_transposed

        output_accumulator.append(np.where(full_out < 1e-7, np.nan, full_out))

    output_array = np.nanmean(np.stack(output_accumulator), axis=0)
    final_cropped = output_array[:, patch_size * scaling_factor:rows * scaling_factor + patch_size * scaling_factor,
                                 patch_size * scaling_factor:cols * scaling_factor + patch_size * scaling_factor]

    super_res_image = final_cropped.transpose(1, 2, 0)  # (H, W, bands)

    if save_path:
        # Update the geo_transform to reflect the new pixel size
        new_transform = geo_transform * geo_transform.scale(1 / scaling_factor, 1 / scaling_factor)

        with rasterio.open(
            save_path,
            "w",
            driver="GTiff",
            height=super_res_image.shape[0],
            width=super_res_image.shape[1],
            count=bands,
            dtype=super_res_image.dtype,
            crs=crs,
            transform=new_transform,
        ) as dst:
            for i in range(bands):
                dst.write(super_res_image[:, :, i], i + 1)


    return super_res_image
