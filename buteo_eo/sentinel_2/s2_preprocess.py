"""
Turn a Sentinel 2 file into a ML dataset.
"""

# standard
import os
import sys; sys.path.append("../../") # Path: buteo_eo/sentinel_2/s2_preprocess.py
from glob import glob

# external
import numpy as np
from osgeo import gdal
from buteo.raster.resample import resample_raster


# internal
from buteo_eo.ai.patch_extraction import extract_patches
from buteo_eo.sentinel_2.s2_utils import get_band_paths


def s2_ready_ml(
    s2_path_file,
    outdir, *,
    patch_size=32,
    overlaps=False,
    resample_20m_to_10m=False,
    resample_alg="bilinear",
):
    paths = get_band_paths(s2_path_file)

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

    if resample_20m_to_10m:
        print("Resampling 20m to 10m..")
        paths_20m_resampled_to_10m = resample_raster(
            paths_20m,
            target_size=paths_10m[0],
            resample_alg="bilinear",
            dst_nodata=None,
            dtype="uint16",
            postfix="",
        )

        paths_10m = paths_10m + paths_20m_resampled_to_10m

    print("Extracting 10m patches..")
    patches = []
    for idx, path in enumerate(paths_10m):
        if idx < 4: continue
        print(f"Extracting patches from {os.path.basename(path)}")
        patch = extract_patches(path, outdir, 32, options={ "overlaps": overlaps })
        patches.append(patch)

    print("Saving 10m patches..")
    if resample_20m_to_10m:
        for resampled_img in paths_20m_resampled_to_10m:
            gdal.Unlink(resampled_img)

        np.savez_compressed(
            outdir + outname_10m,
            B02=np.load(patches[0]),
            B03=np.load(patches[1]),
            B04=np.load(patches[2]),
            B05=np.load(patches[4]),
            B06=np.load(patches[5]),
            B07=np.load(patches[6]),
            B08=np.load(patches[3]),
            B8A=np.load(patches[7]),
            B11=np.load(patches[8]),
            B12=np.load(patches[9]),
            SCL=np.load(patches[10]),
        )
    else:
        np.savez_compressed(
            outdir + outname_10m,
            B02=np.load(patches[0]),
            B03=np.load(patches[1]),
            B04=np.load(patches[2]),
            B08=np.load(patches[3]),
        )

    print("Cleaning 10m temporary files..")
    for file in patches:
        os.remove(file)

    import pdb; pdb.set_trace()


    if not resample_20m_to_10m:
        print("Extracting 20m patches..")
        patches = extract_patches(
            paths_20m,
            outdir,
            16,
            options={
                "overlaps": overlaps,
            },
        )

        print("Saving 20m patches..")
        np.savez_compressed(
            outdir + outname_20m,
            B05=np.load(patches[0]),
            B06=np.load(patches[1]),
            B07=np.load(patches[2]),
            B8A=np.load(patches[3]),
            B11=np.load(patches[4]),
            B12=np.load(patches[5]),
            SCL=np.load(patches[6]),
        )

        print("Cleaning 20m temporary files..")
        for file in patches:
            os.remove(file)

        return outname_10m, outname_20m

    else:
        return (outname_10m)



if __name__ == "__main__":
    s2_path = "/home/casper/Desktop/data/sentinel2_images/"
    safe = glob(s2_path + "*.SAFE")[0]
    s2_ready_ml(safe, s2_path, resample_20m_to_10m=True)
