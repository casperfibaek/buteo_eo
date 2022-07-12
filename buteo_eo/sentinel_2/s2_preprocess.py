"""
Turn a Sentinel 2 file into a ML dataset.
"""

# standard
import os
import sys; sys.path.append("../../") # Path: buteo_eo/sentinel_2/s2_preprocess.py
from concurrent.futures import ThreadPoolExecutor

# external
import numpy as np
from osgeo import gdal
from buteo.raster.resample import resample_raster
from buteo.filters.norm_rasters import norm_to_range

# internal
from buteo_eo.ai.patch_extraction import extract_patches
from buteo_eo.sentinel_2.s2_utils import get_band_paths


def normalise_s2_arr(
    arr,
    method="max_truncate",
    target_max_value=1.0,
    target_min_value=0.0,
    original_max_value=10000.0,
    original_min_value=0.0,
    force_float32=True,
):
    if method == "max_truncate":
        arr = norm_to_range(
            arr,
            target_min_value,
            target_max_value,
            original_min_value,
            original_max_value,
            truncate=True,
        )
    elif method == "max":
        arr = norm_to_range(
            arr,
            target_min_value,
            target_max_value,
            0,
            original_max_value,
            truncate=False,
        )
    elif method == "min_max":
        arr = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    elif method == "standardise":
        arr = (arr - np.nanmean(arr)) / np.nanstd(arr)
    elif method == "standardise_mad":
        median = np.nanmedian(arr)
        mad = np.nanmedian(np.abs(arr - median))
        mad_std = mad * 1.4826
        arr = (arr - median) / mad_std
    else:
        raise ValueError(f"Unknown normalise method: {method}, must be one of: max_truncate, max, min_max, standardise")
    
    return arr.astype(np.float32) if force_float32 else arr


def s2_ready_ml(
    s2_path_file,
    outdir, *,
    process_20m=True,
    patch_size_10m=32,
    overlaps=False,
    resample_20m_to_10m=False,
    resample_alg="bilinear",
    use_multithreading=True,
    normalise=False,
    normalise_method="max_truncate",
    normalise_target_max_value=1.0,
    normalise_target_min_value=0.0,
    normalise_original_max_value=10000.0,
    normalise_original_min_value=0.0,
    normalise_force_float32=True,
    clean=True,
):
    paths = get_band_paths(s2_path_file)

    assert patch_size_10m % 2 == 0, "Patch size must be even"
    assert outdir is not None, "Output directory must be specified"
    assert os.path.isdir(outdir), "Output directory must exist"

    # 10 meter bands
    paths_10m = [
        paths["10m"]["B02"],
        paths["10m"]["B03"],
        paths["10m"]["B04"],
        paths["10m"]["B08"],
    ]

    # 20 meter bands
    paths_20m = [
        paths["20m"]["B05"],
        paths["20m"]["B06"],
        paths["20m"]["B07"],
        paths["20m"]["B8A"],
        paths["20m"]["B11"],
        paths["20m"]["B12"],
        paths["20m"]["SCL"],
    ]

    outname_10m = "_".join(os.path.basename(paths["10m"]["B02"]).split("_")[:-2]) + "_10m.npz"
    outname_20m = "_".join(os.path.basename(paths["20m"]["B05"]).split("_")[:-2]) + "_20m.npz"

    patches = []

    if use_multithreading:

        length = len(paths_10m) + len(paths_20m) if resample_20m_to_10m and process_20m else len(paths_10m)

        if resample_20m_to_10m and process_20m:

            print("Resampling 20m to 10m..")
            paths_20m_resampled_to_10m = resample_raster(
                paths_20m,
                target_size=paths_10m[0],
                resample_alg=resample_alg,
                dst_nodata=None,
                dtype="uint16",
                postfix="",
            )
            paths_10m += paths_20m_resampled_to_10m

        print("Extracting 10m patches from images..")

        kwargs = []
        for idx, path in enumerate(paths_10m):
            kwargs.append({
                "raster_list": path,
                "outdir": outdir,
                "patch_size": patch_size_10m,
                "overlaps": overlaps,
                "thread_id": idx,
            })

        with ThreadPoolExecutor(max_workers=length) as executor:
            patches += executor.map(lambda x: extract_patches(**x), kwargs)

    else:
        for _idx, path in enumerate(paths_10m):
            print(f"Extracting patches from {os.path.basename(path)}")
            patch = extract_patches(
                path,
                outdir,
                patch_size=patch_size_10m,
                overlaps=overlaps,
            )
            patches.append(patch)

    def _norm(arr_file):
        return normalise_s2_arr(
            np.load(arr_file),
            method=normalise_method,
            target_max_value=normalise_target_max_value,
            target_min_value=normalise_target_min_value,
            original_max_value=normalise_original_max_value,
            original_min_value=normalise_original_min_value,
            force_float32=normalise_force_float32
        )

    print("Saving 10m patches..")
    if resample_20m_to_10m and process_20m:
        for resampled_img in paths_20m_resampled_to_10m:
            gdal.Unlink(resampled_img)

        np.savez_compressed(
            outdir + outname_10m,
            B02=np.load(patches[0]) if not normalise else _norm(patches[0]),
            B03=np.load(patches[1]) if not normalise else _norm(patches[1]),
            B04=np.load(patches[2]) if not normalise else _norm(patches[2]),
            B05=np.load(patches[4]) if not normalise else _norm(patches[4]),
            B06=np.load(patches[5]) if not normalise else _norm(patches[5]),
            B07=np.load(patches[6]) if not normalise else _norm(patches[6]),
            B08=np.load(patches[3]) if not normalise else _norm(patches[3]),
            B8A=np.load(patches[7]) if not normalise else _norm(patches[7]),
            B11=np.load(patches[8]) if not normalise else _norm(patches[8]),
            B12=np.load(patches[9]) if not normalise else _norm(patches[9]),
            SCL=np.load(patches[10]) if not normalise else _norm(patches[10]),
        )
    else:
        np.savez_compressed(
            outdir + outname_10m,
            B02=np.load(patches[0]) if not normalise else _norm(patches[0]),
            B03=np.load(patches[1]) if not normalise else _norm(patches[1]),
            B04=np.load(patches[2]) if not normalise else _norm(patches[2]),
            B08=np.load(patches[3]) if not normalise else _norm(patches[3]),
        )

    if clean:
        print("Cleaning 10m temporary files..")
        for file in patches:
            os.remove(file)

    if not resample_20m_to_10m and process_20m:
        print("Extracting 20m patches..")
        patches = []

        if use_multithreading:

            kwargs = []
            for idx, path in enumerate(paths_20m):
                kwargs.append({
                    "raster_list": path,
                    "outdir": outdir,
                    "patch_size": patch_size_10m // 2,
                    "overlaps": overlaps,
                    "thread_id": idx,
                })

            with ThreadPoolExecutor(max_workers=len(paths_20m)) as executor:
                patches += executor.map(lambda x: extract_patches(**x), kwargs)

        else:
            for _idx, path in enumerate(paths_20m):
                print(f"Extracting patches from {os.path.basename(path)}")
                
                patch = extract_patches(
                    path,
                    outdir,
                    patch_size=patch_size_10m // 2,
                    overlaps=overlaps,
                )
                patches.append(patch)

        print("Saving 20m patches..")
        np.savez_compressed(
            outdir + outname_20m,
            B05=np.load(patches[0]) if not normalise else _norm(patches[0]),
            B06=np.load(patches[1]) if not normalise else _norm(patches[1]),
            B07=np.load(patches[2]) if not normalise else _norm(patches[2]),
            B8A=np.load(patches[3]) if not normalise else _norm(patches[3]),
            B11=np.load(patches[4]) if not normalise else _norm(patches[4]),
            B12=np.load(patches[5]) if not normalise else _norm(patches[5]),
            SCL=np.load(patches[6]) if not normalise else _norm(patches[6]),
        )

        if clean:
            print("Cleaning 20m temporary files..")
            for file in patches:
                os.remove(file)

        return outdir + outname_10m, outdir + outname_20m

    else:
        return (outdir + outname_10m)


if __name__ == "__main__":
    from glob import glob
    s2_path = "/home/casper/Desktop/data/sentinel2_images/"
    safe = glob(s2_path + "*.SAFE")[0]
    # s2_ready_ml(safe, s2_path, resample_20m_to_10m=False, process_20m=True)
    bob = s2_ready_ml(safe, s2_path, resample_20m_to_10m=False, process_20m=True, normalise=True, normalise_original_max_value=10000.0)
    import pdb; pdb.set_trace()