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


def normalise_s2_ml(npz_file):
    loaded = np.load(npz_file)
    files = loaded.files

    normed = {}

    for file in files:
        if file == "SCL":
            normed[file] = loaded[file]
            continue

        print(f"Normalising: {file}")
        normed[file] = normalise_s2_arr(loaded[file])
    
    outname = os.path.join(os.path.dirname(npz_file), os.path.splitext(os.path.basename(npz_file))[0] + "_normed.npz")
    np.savez(outname, **normed)


# TODO
def augment_s2_arr(arr):
    return arr


def s2_ready_ml(
    s2_path_file,
    outdir,
    *,
    patch_size=32,
    process_20m=True,
    offsets=True,
    offset_count=3,
    include_downsampled_bands=False,
    resample_20m_to_10m=False,
    resample_alg="bilinear",
    process_bands=None,
    aoi_mask=None,
    aoi_mask_tolerance=0.0, # 0 means no tolerance for pixels outside of AIO mask
    aoi_mask_layerid=0,
    aoi_mask_output=False,
    labels=None,
    labels_attribute="class", # None creates a binary mask of 0-1
    labels_baseline=0,
    labels_resolution="match_10m",
    labels_fuzz_from=False,
    labels_fuzz=False,
    use_multithreading=True,
    tmpdir=None,
    clean=True,
):
    """
    This function takes a Sentinel 2 file and returns a ML dataset.

    Args:
        s2_path_file: Path to Sentinel 2 file.
        outdir: Output directory.

    Keyword Args:
        patch_size: Size of patches to extract.
        process_20m: Whether to process 20m bands.
        offsets: Whether to extract offsets.
        offset_count: Number of offsets to extract. (0-9)
        resample_20m_to_10m: Whether to resample 20m bands to 10m.
        resample_alg: Resampling algorithm to use.
        aoi_mask: Path to AOI mask.
        aoi_mask_tolerance: Tolerance for pixels outside of AOI mask.
        use_multithreading: Whether to use multithreading.
        include_downsampled_bands: Whether to include downsampled bands.
        labels: Path to labels.
        labels_attribute: Attribute to use for labels.
        labels_baseline: Baseline value for labels.
        labels_resolution: Resolution to use for labels.
        labels_fuzz_from: Whether to fuzz labels from the original file.
        clean: Whether to clean up temporary files.
    """
    paths = get_band_paths(s2_path_file)

    assert patch_size % 2 == 0, "Patch size must be even"
    assert outdir is not None, "Output directory must be specified"
    assert os.path.isdir(outdir), "Output directory must exist"
    assert aoi_mask_tolerance >= 0.0 and aoi_mask_tolerance <= 1.0, "AOI mask tolerance must be between 0 and 1"

    if include_downsampled_bands and (not process_20m or resample_20m_to_10m):
        raise ValueError("Cannot include downsampled bands if not processing 20m bands. Cannot resample 20m to 10m and include downsampled bands.")

    if aoi_mask_output and aoi_mask is None:
        raise ValueError("AOI mask output requires an AOI mask")

    if tmpdir is None:
        tmpdir = outdir

    all_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12", "SCL"]
    if process_bands == "all" or process_bands is None or process_bands is False or len(process_bands) == 0:
        process_bands = all_bands
    else:
        for band in process_bands:
            assert band in all_bands, f"Band {band} is not a valid band"

    # 10 meter bands
    paths_10m = [
        { "name": "B02", "path": paths["10m"]["B02"], "size": 10 },
        { "name": "B03", "path": paths["10m"]["B03"], "size": 10 },
        { "name": "B04", "path": paths["10m"]["B04"], "size": 10 },
        { "name": "B08", "path": paths["10m"]["B08"], "size": 10 },
    ]

    # 20 meter bands
    paths_20m_downsampled = [
        { "name": "B02", "path": paths["20m"]["B02"], "size": 20 },
        { "name": "B03", "path": paths["20m"]["B03"], "size": 20 },
        { "name": "B04", "path": paths["20m"]["B04"], "size": 20 },
        # paths["20m"]["B08"], not included in 20m. Will be resampled from 10m
    ]

    paths_20m = [
        { "name": "B05", "path": paths["20m"]["B05"], "size": 20 },
        { "name": "B06", "path": paths["20m"]["B06"], "size": 20 },
        { "name": "B07", "path": paths["20m"]["B07"], "size": 20 },
        { "name": "B8A", "path": paths["20m"]["B8A"], "size": 20 },
        { "name": "B11", "path": paths["20m"]["B11"], "size": 20 },
        { "name": "B12", "path": paths["20m"]["B12"], "size": 20 },
        { "name": "SCL", "path": paths["20m"]["SCL"], "size": 20 },
    ]

    if include_downsampled_bands:
        b08_resampled = resample_raster(
            paths["10m"]["B08"],
            target_size=paths["20m"]["B05"],
            resample_alg="average",
            postfix="_10m_to_20m",
            dtype="uint16"
        )
        paths_20m_downsampled.append({"name": "B08", "path": b08_resampled, "size": 20})
        paths_20m = paths_20m + paths_20m_downsampled

    outname_10m = "_".join(os.path.basename(paths["10m"]["B02"]).split("_")[:-2]) + "_10m.npz"
    outname_20m = "_".join(os.path.basename(paths["20m"]["B05"]).split("_")[:-2]) + "_20m.npz"

    outname_10m_masks = "_".join(os.path.basename(paths["10m"]["B02"]).split("_")[:-2]) + "_10m_masks.npz"
    outname_20m_masks = "_".join(os.path.basename(paths["20m"]["B05"]).split("_")[:-2]) + "_20m_masks.npz"

    patches = {}
    patches_masks = {}

    if use_multithreading:

        length = len(paths_10m) + len(paths_20m) if resample_20m_to_10m and process_20m else len(paths_10m)

        if resample_20m_to_10m and process_20m:

            print("Resampling 20m to 10m..")
            paths_to_resample = [d["path"] for d in paths_20m if d["name"] in process_bands]

            if len(paths_to_resample) == 0:
                raise ValueError("No bands selected to resample. Please add 20m bands to process_bands or set process_20m to False.")

            paths_20m_resampled_to_10m = resample_raster(
                paths_to_resample,
                target_size=paths_10m[0]["path"],
                resample_alg=resample_alg,
                dst_nodata=None,
                dtype="uint16",
                postfix="",
            )

            for i, path in enumerate(paths_20m_resampled_to_10m):
                paths_20m[i]["path"] = path
                paths_20m[i]["size"] = 10

            paths_10m += paths_20m

        print("Extracting 10m patches from images..")

        kwargs = []
        for idx, band in enumerate(paths_10m):

            if band["name"] not in process_bands:
                continue

            kwargs.append({
                "raster_list": band["path"],
                "outdir": tmpdir,
                "patch_size": patch_size,
                "offsets": offsets,
                "offset_count": offset_count,
                "aoi_mask": aoi_mask,
                "aoi_mask_tolerance": aoi_mask_tolerance,
                "aoi_mask_layerid": aoi_mask_layerid,
                "aoi_mask_output": aoi_mask_output,
                "thread_id": idx,
                "thread_name": band["name"],
            })

        with ThreadPoolExecutor(max_workers=length) as executor:
            results = executor.map(lambda x: extract_patches(**x), kwargs)
        
        for result in results:
            thread_arr, thread_mask, thread_name = result

            patches[thread_name] = thread_arr
            patches_masks[thread_name] = thread_mask

    else:
        for _idx, band in enumerate(paths_10m):

            if band["name"] not in process_bands:
                continue

            print(f"Extracting patches from {os.path.basename(band['path'])}")
            results = extract_patches(
                raster_list=band["path"],
                outdir=tmpdir,
                patch_size=patch_size,
                offsets=offsets,
                offset_count=offset_count,
                aoi_mask=aoi_mask,
                aoi_mask_tolerance=aoi_mask_tolerance,
                aoi_mask_layerid=aoi_mask_layerid,
                aoi_mask_output=aoi_mask_output,
            )

            thread_arr, thread_mask, _thread_name = results

            patches[band["name"]] = thread_arr
            patches_masks[band["name"]] = thread_mask

    print("Saving 10m patches..")
    if resample_20m_to_10m and process_20m:
        for resampled_img in paths_20m_resampled_to_10m:
            gdal.Unlink(resampled_img)

        np.savez_compressed(
            outdir + outname_10m,
            B02=np.load(patches["B02"]) if "B02" in process_bands else None,
            B03=np.load(patches["B03"]) if "B03" in process_bands else None,
            B04=np.load(patches["B04"]) if "B04" in process_bands else None,
            B05=np.load(patches["B05"]) if "B05" in process_bands else None,
            B06=np.load(patches["B06"]) if "B06" in process_bands else None,
            B07=np.load(patches["B07"]) if "B07" in process_bands else None,
            B08=np.load(patches["B08"]) if "B08" in process_bands else None,
            B8A=np.load(patches["B8A"]) if "B8A" in process_bands else None,
            B11=np.load(patches["B11"]) if "B11" in process_bands else None,
            B12=np.load(patches["B12"]) if "B12" in process_bands else None,
            SCL=np.load(patches["SCL"]) if "SCL" in process_bands else None,
        )

        if aoi_mask_output:
            np.savez_compressed(
                outdir + outname_10m_masks,
                B02=np.load(patches_masks["B02"]) if "B02" in process_bands else None,
                B03=np.load(patches_masks["B03"]) if "B03" in process_bands else None,
                B04=np.load(patches_masks["B04"]) if "B04" in process_bands else None,
                B05=np.load(patches_masks["B05"]) if "B05" in process_bands else None,
                B06=np.load(patches_masks["B06"]) if "B06" in process_bands else None,
                B07=np.load(patches_masks["B07"]) if "B07" in process_bands else None,
                B08=np.load(patches_masks["B08"]) if "B08" in process_bands else None,
                B8A=np.load(patches_masks["B8A"]) if "B8A" in process_bands else None,
                B11=np.load(patches_masks["B11"]) if "B11" in process_bands else None,
                B12=np.load(patches_masks["B12"]) if "B12" in process_bands else None,
                SCL=np.load(patches_masks["SCL"]) if "SCL" in process_bands else None,
            )

    else:

        np.savez_compressed(
            outdir + outname_10m,
            B02=np.load(patches["B02"]) if "B02" in process_bands else None,
            B03=np.load(patches["B03"]) if "B03" in process_bands else None,
            B04=np.load(patches["B04"]) if "B04" in process_bands else None,
            B08=np.load(patches["B08"]) if "B08" in process_bands else None,
        )

        if aoi_mask_output:
            np.savez_compressed(
                outdir + outname_10m_masks,
                B02=np.load(patches_masks["B02"]) if "B02" in process_bands else None,
                B03=np.load(patches_masks["B03"]) if "B03" in process_bands else None,
                B04=np.load(patches_masks["B04"]) if "B04" in process_bands else None,
                B08=np.load(patches_masks["B08"]) if "B08" in process_bands else None,
            )

    if clean:
        print("Cleaning 10m temporary files..")
        for band in process_bands:
            if band in patches:
                os.remove(patches[band])

                if aoi_mask_output:
                    os.remove(patches_masks[band])


    if not resample_20m_to_10m and process_20m:
        print("Extracting 20m patches..")

        patches = {}
        patches_masks = {}

        if use_multithreading:

            kwargs = []
            for idx, band in enumerate(paths_20m):

                if band["name"] not in process_bands:
                    continue

                kwargs.append({
                    "raster_list": band["path"],
                    "outdir": tmpdir,
                    "patch_size": patch_size // 2,
                    "offsets": offsets,
                    "offset_count": offset_count,
                    "aoi_mask": aoi_mask,
                    "aoi_mask_tolerance": aoi_mask_tolerance,
                    "aoi_mask_layerid": aoi_mask_layerid,
                    "aoi_mask_output": aoi_mask_output,
                    "thread_id": idx,
                    "thread_name": band["name"],
                })

            with ThreadPoolExecutor(max_workers=len(paths_20m)) as executor:
                results = executor.map(lambda x: extract_patches(**x), kwargs)

            for result in results:
                thread_arr, thread_mask, thread_name = result

                patches[thread_name] = thread_arr
                patches_masks[thread_name] = thread_mask

        else:
            for _idx, band in enumerate(paths_20m):

                if band["name"] not in process_bands:
                    continue

                print(f"Extracting patches from {os.path.basename(band['path'])}")
                
                results = extract_patches(
                    raster_list=band["path"],
                    outdir=tmpdir,
                    patch_size=patch_size // 2,
                    offsets=offsets,
                    offset_count=offset_count,
                    aoi_mask=aoi_mask,
                    aoi_mask_tolerance=aoi_mask_tolerance,
                    aoi_mask_layerid=aoi_mask_layerid,
                    aoi_mask_output=aoi_mask_output,
                )

                thread_arr, thread_mask, _thread_name = results

                patches[band["name"]] = thread_arr
                patches_masks[band["name"]] = thread_mask

        print("Saving 20m patches..")
        if include_downsampled_bands:
            gdal.Unlink(b08_resampled)
            np.savez_compressed(
                outdir + outname_20m,
                B02=np.load(patches["B02"]) if "B02" in process_bands else None,
                B03=np.load(patches["B03"]) if "B03" in process_bands else None,
                B04=np.load(patches["B04"]) if "B04" in process_bands else None,
                B05=np.load(patches["B05"]) if "B05" in process_bands else None,
                B06=np.load(patches["B06"]) if "B06" in process_bands else None,
                B07=np.load(patches["B07"]) if "B07" in process_bands else None,
                B08=np.load(patches["B08"]) if "B08" in process_bands else None,
                B8A=np.load(patches["B8A"]) if "B8A" in process_bands else None,
                B11=np.load(patches["B11"]) if "B11" in process_bands else None,
                B12=np.load(patches["B12"]) if "B12" in process_bands else None,
                SCL=np.load(patches["SCL"]) if "SCL" in process_bands else None,
            )

            if aoi_mask_output:
                np.savez_compressed(
                    outdir + outname_20m_masks,
                    B02=np.load(patches_masks["B02"]) if "B02" in process_bands else None,
                    B03=np.load(patches_masks["B03"]) if "B03" in process_bands else None,
                    B04=np.load(patches_masks["B04"]) if "B04" in process_bands else None,
                    B05=np.load(patches_masks["B05"]) if "B05" in process_bands else None,
                    B06=np.load(patches_masks["B06"]) if "B06" in process_bands else None,
                    B07=np.load(patches_masks["B07"]) if "B07" in process_bands else None,
                    B08=np.load(patches_masks["B08"]) if "B08" in process_bands else None,
                    B8A=np.load(patches_masks["B8A"]) if "B8A" in process_bands else None,
                    B11=np.load(patches_masks["B11"]) if "B11" in process_bands else None,
                    B12=np.load(patches_masks["B12"]) if "B12" in process_bands else None,
                    SCL=np.load(patches_masks["SCL"]) if "SCL" in process_bands else None,
                )
        else:
            np.savez_compressed(
                outdir + outname_20m,
                B05=np.load(patches["B05"]) if "B05" in process_bands else None,
                B06=np.load(patches["B06"]) if "B06" in process_bands else None,
                B07=np.load(patches["B07"]) if "B07" in process_bands else None,
                B8A=np.load(patches["B8A"]) if "B8A" in process_bands else None,
                B11=np.load(patches["B11"]) if "B11" in process_bands else None,
                B12=np.load(patches["B12"]) if "B12" in process_bands else None,
                SCL=np.load(patches["SCL"]) if "SCL" in process_bands else None,
            )

            if aoi_mask_output:
                np.savez_compressed(
                    outdir + outname_20m_masks,                   
                    B05=np.load(patches_masks["B05"]) if "B05" in process_bands else None,
                    B06=np.load(patches_masks["B06"]) if "B06" in process_bands else None,
                    B07=np.load(patches_masks["B07"]) if "B07" in process_bands else None,
                    B8A=np.load(patches_masks["B8A"]) if "B8A" in process_bands else None,
                    B11=np.load(patches_masks["B11"]) if "B11" in process_bands else None,
                    B12=np.load(patches_masks["B12"]) if "B12" in process_bands else None,
                    SCL=np.load(patches_masks["SCL"]) if "SCL" in process_bands else None,
                )

        if clean:
            print("Cleaning 20m temporary files..")
            for band in process_bands:
                if band in patches:
                    os.remove(patches[band])

                    if aoi_mask_output:
                        os.remove(patches_masks[band])

    return (
        outdir + outname_10m,
        outdir + outname_10m_masks if aoi_mask_output else None,
        outdir + outname_20m,
        outdir + outname_20m_masks if aoi_mask_output else None,
    )


if __name__ == "__main__":
    from glob import glob
    s2_path = "/home/casper/Desktop/data/sentinel2_images/"
    tmpdir = "/home/casper/Desktop/data/sentinel2_images/tmp/"
    beirut = "/home/casper/Desktop/data/beirut_boundary.gpkg"
    safe = glob(s2_path + "*.SAFE")[0]
    # s2_ready_ml(safe, s2_path, resample_20m_to_10m=False, process_20m=True)
    path_arr_10m, path_arr_10m_mask, path_arr_20m, path_arr_20m_mask = s2_ready_ml(
        safe,
        s2_path,
        aoi_mask=beirut,
        aoi_mask_tolerance=0.0,
        aoi_mask_output=False,
        process_bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"],
        tmpdir=tmpdir,
    )

    normalise_s2_ml(path_arr_10m)
    normalise_s2_ml(path_arr_20m)

    import pdb; pdb.set_trace()
