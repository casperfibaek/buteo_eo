"""
This module generates patches/tiles from a raster.

TODO:
    - Improve documentation
    - Explain options
"""

# standard
from concurrent.futures import thread
import os
import sys; sys.path.append("../../") # Path: buteo_eo/ai/patch_extraction.py
from uuid import uuid4

# external
import numpy as np
from osgeo import ogr, gdal
from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
    is_raster,
    raster_to_metadata,
    stack_rasters_vrt,
)
from buteo.raster.align import rasters_are_aligned, align_rasters
from buteo.raster.clip import clip_raster
from buteo.raster.resample import resample_raster
from buteo.vector.io import vector_to_metadata, is_vector, open_vector
from buteo.vector.attributes import vector_get_fids
from buteo.vector.rasterize import rasterize_vector
from buteo.vector.reproject import reproject_vector
from buteo.utils.core import progress

# internal
from buteo_eo.ai.ml_utils import get_offsets
from buteo_eo.ai.patch_utils import get_overlaps


def extract_patches(
    raster_list,
    outdir,
    patch_size=32,
    offsets=True,
    offset_count=3,
    border_check=True,
    force_align=False,
    aoi_mask=None,
    aoi_mask_tolerance=0.0,
    aoi_mask_layerid=0,
    aoi_mask_output=False,
    aoi_mask_value=0,
    prefix="",
    postfix="",
    thread_id="",
    thread_name="",
    verbose=False,
):
    """
    Generate patches for machine learning from rasters
    """

    if aoi_mask is not None and not is_vector(aoi_mask):
        raise TypeError("Clip geom is invalid. Did you input a valid geometry?")

    input_was_single_raster = False
    if not isinstance(raster_list, list):
        raster_list = [raster_list]
        input_was_single_raster = True

    for raster in raster_list:
        if not is_raster(raster):
            raise TypeError("raster_list is not a list of rasters.")

    if not os.path.isdir(outdir):
        raise ValueError(
            "Outdir does not exist. Please create before running the function."
        )

    if not rasters_are_aligned(raster_list, same_extent=True):
        if force_align:
            print(
                "Rasters were not aligned. Realigning rasters due to force_align=True option."
            )
            raster_list = align_rasters(raster_list, postfix="")
        else:
            raise ValueError("Rasters in raster_list are not aligned. Set force_align=True if you want to realign rasters.")

    offsets = get_offsets(patch_size, number_of_offsets=offset_count) if offsets else [[0, 0]]
    raster_metadata = raster_to_metadata(raster_list[0], create_geometry=True)

    if aoi_mask is None:
        aoi_mask = raster_metadata["extent_datasource_path"]
    elif not is_vector(aoi_mask) and is_raster(aoi_mask):
        aoi_mask = raster_to_metadata(aoi_mask, create_geometry=True)["extent_datasource_path"]
    
    aoi_mask = reproject_vector(aoi_mask, raster_metadata["projection"], prefix=str(thread_id) + "_", copy_if_same=True)

    zones_meta = vector_to_metadata(aoi_mask)

    mem_driver = ogr.GetDriverByName("ESRI Shapefile")

    if zones_meta["layer_count"] == 0:
        raise ValueError("Vector contains no layers.")

    zones_layer_meta = zones_meta["layers"][aoi_mask_layerid]

    if zones_layer_meta["geom_type"] not in ["Multi Polygon", "Polygon"]:
        raise ValueError("clip geom is not Polygon or Multi Polygon.")

    zones_ogr = open_vector(aoi_mask)
    zones_layer = zones_ogr.GetLayer(aoi_mask_layerid)
    feature_defn = zones_layer.GetLayerDefn()
    fids = vector_get_fids(zones_ogr, aoi_mask_layerid)

    if verbose:
        progress(0, len(fids) * len(raster_list), "processing fids")

    processed_fids = []
    processed = 0

    outputs = []
    outputs_masks = []

    for idx, raster in enumerate(raster_list):
        name = os.path.splitext(os.path.basename(raster))[0]
        list_extracted = []
        list_masks = []

        for fid in fids:
            path_valid =    f"/vsimem/{thread_id}_{uuid4().int}_{str(idx)}_{str(fid)}_valid.tif"
            path_fid =      f"/vsimem/{thread_id}_{uuid4().int}_{str(idx)}_{str(fid)}_fid.shp"
            path_extent =   f"/vsimem/{thread_id}_{uuid4().int}_{str(idx)}_{str(fid)}_extent.tif"
            path_clip =     f"/vsimem/{thread_id}_{uuid4().int}_{str(idx)}_{str(fid)}_clip.tif"

            feature = zones_layer.GetFeature(fid)
            geom = feature.GetGeometryRef()
            fid_ds = mem_driver.CreateDataSource(path_fid)
            fid_ds_lyr = fid_ds.CreateLayer(
                "fid_layer",
                geom_type=ogr.wkbPolygon,
                srs=zones_layer_meta["projection_osr"],
            )
            copied_feature = ogr.Feature(feature_defn)
            copied_feature.SetGeometry(geom)
            fid_ds_lyr.CreateFeature(copied_feature)

            fid_ds.FlushCache()
            fid_ds.SyncToDisk()

            extent = clip_raster(
                raster,
                clip_geom=path_fid,
                out_path=path_extent,
                adjust_bbox=True,
                all_touch=False,
                to_extent=True,
            )

            rasterize_vector(
                path_fid,
                (raster_metadata["pixel_width"], raster_metadata["pixel_height"]),
                out_path=path_valid,
                extent=extent,
            )

            valid_arr = raster_to_array(path_valid)

            try:
                clip_raster(
                    raster,
                    clip_geom=path_valid,
                    out_path=path_clip,
                    all_touch=False,
                    adjust_bbox=False,
                )
            except Exception as error_message:
                print(f"Warning: {raster} did not intersect geom with fid: {fid}.")
                print(error_message)

                continue

            arr = raster_to_array(path_clip)

            gdal.Unlink(extent)
            gdal.Unlink(path_fid)
            gdal.Unlink(path_valid)
            gdal.Unlink(path_clip)

            if arr.shape[:2] != valid_arr.shape[:2]:
                raise Exception(
                    f"Error while matching array shapes. Raster: {arr.shape}, Valid: {valid_arr.shape}"
                )

            arr_offsets = get_overlaps(arr, offsets, patch_size, border_check)

            arr = np.concatenate(arr_offsets)
            valid_offsets = np.concatenate(
                get_overlaps(valid_arr, offsets, patch_size, border_check)
            )

            valid_mask = (
                (1 - (valid_offsets.sum(axis=(1, 2)) / (patch_size * patch_size)))
                <= aoi_mask_tolerance
            )[:, 0]

            arr = arr[valid_mask]
            valid_masked = valid_offsets[valid_mask]
            
            list_extracted.append(arr)
            list_masks.append(valid_masked)
                
            if fid not in processed_fids:
                processed_fids.append(fid)

            processed += 1

            if verbose:
                progress(processed, len(fids) * len(raster_list), "processing fids")


        out_arr = np.ma.concatenate(list_extracted).filled(aoi_mask_value)
        out_path = f"{outdir}{prefix}{name}{postfix}.npy"
        outputs.append(out_path)

        np.save(out_path, out_arr)

        if aoi_mask_output:
            out_mask_path = f"{outdir}{prefix}{name}_mask{postfix}.npy"

            out_mask = np.ma.concatenate(list_masks).filled(aoi_mask_value)
            outputs_masks.append(out_mask_path)

            np.save(out_mask_path, out_mask)
        else:
            outputs_masks.append(None)

    if aoi_mask is not None:
        gdal.Unlink(aoi_mask)

    if input_was_single_raster:
        return outputs[0], outputs_masks[0], thread_name
    
    return outputs, outputs_masks, [thread_name] * len(outputs)


from glob import glob
folder = "/home/casper/Desktop/dataset_test/"

bands = glob(folder + "B*_10m.npy")
loaded = []
for band in bands:
    load = np.load(band)
    loaded.append((np.where(load == 65535, 0, load) / 10000.0).astype(np.float32))

stacked_bands = np.concatenate(loaded, axis=3)

labels = glob(folder + "labels*_10m.npy")
loaded = []
for label in labels:
    load = np.load(label)
    loaded.append(np.where(load < 0, 0, load))

stacked_labels = np.concatenate(loaded, axis=3)

np.savez_compressed(folder + "roads_builds_bornholm.npz", bands=stacked_bands, labels=stacked_labels)


# images = glob(folder + "*_10m.tif")
# extract_patches(
#     images,
#     patch_size=128,
#     offset_count=9,
#     outdir=folder,
#     aoi_mask=folder + "mask.gpkg"
# )


def rasterize_labels(
    geom,
    reference,
    *,
    class_attrib=None,
    out_path=None,
    resample_from=None,
    resample_to=None,
    resample_alg="average",
    resample_scale=None,
    align=True,
    dtype="float32",
    ras_dtype="uint8",
):
    if not is_vector(geom):
        raise TypeError(
            "label geom is invalid. Did you input a valid geometry?"
        )

    if resample_from is None and resample_to is None:
        rasterized = rasterize_vector(
            geom,
            reference,
            extent=reference,
            attribute=class_attrib,
            dtype=ras_dtype,
        )
    elif resample_from is not None and resample_to is None:
        rasterized_01 = rasterize_vector(
            geom,
            resample_from,
            extent=reference,
            attribute=class_attrib,
            dtype=ras_dtype,
        )
        rasterized = resample_raster(
            rasterized_01,
            reference,
            resample_alg=resample_alg,
            dtype=dtype,
            dst_nodata=None,
        )
        gdal.Unlink(rasterized_01)
    elif resample_from is None and resample_to is not None:
        rasterized = rasterize_vector(
            geom,
            resample_to,
            extent=reference,
            attribute=class_attrib,
            dtype=ras_dtype,
        )
    elif resample_from is not None and resample_to is not None:
        rasterized_01 = rasterize_vector(
            geom,
            resample_from,
            extent=reference,
            attribute=class_attrib,
            dtype=ras_dtype,
        )
        rasterized = resample_raster(
            rasterized_01,
            resample_to,
            resample_alg=resample_alg,
            dtype=dtype,
            dst_nodata=None,
        )
        gdal.Unlink(rasterized_01)

    if align:
        aligned = align_rasters(rasterized, master=reference, dst_nodata=None)
        gdal.Unlink(rasterized)
        rasterized = aligned

    arr = raster_to_array(rasterized)
    if isinstance(arr, np.ma.MaskedArray):
        arr.fill_value = 0
        arr = arr.filled(0)

    if resample_scale is not None:
        return array_to_raster(arr * resample_scale, reference=reference, out_path=out_path, set_nodata=None)
    else:
        return array_to_raster(arr, reference=reference, out_path=out_path, set_nodata=None)


def create_mask(geom, reference, out_path=None):
    if not is_vector(geom):
        raise TypeError(
            "label geom is invalid. Did you input a valid geometry?"
        )

    mask = rasterize_vector(
        geom,
        reference,
        out_path=out_path,
        extent=reference,
        attribute=None,
        dtype="uint8",
    )

    return mask


def create_labels(
    geom,
    reference, *,
    out_path=None,
    tmp_folder=None,
    grid=None,
    resample_from=0.2,
    resample_scale=100.0,
    round_label=None,
):
    label_stack = []

    cells = [geom] if grid is None else grid

    for cell in cells:
        name = os.path.splitext(os.path.basename(cell))[0]
        bounds = clip_raster(
            reference,
            cell,
            to_extent=True,
            all_touch=False,
            adjust_bbox=True,
        )

        labels = rasterize_labels(
            geom,
            bounds,
            resample_from=resample_from,
            resample_scale=resample_scale,
        )
        labels_set = array_to_raster(raster_to_array(labels), reference=labels, out_path=tmp_folder + f"{name}_labels_10m.tif", set_nodata=None)
        label_stack.append(labels_set)
        gdal.Unlink(labels)
        gdal.Unlink(bounds)

    stacked = stack_rasters_vrt(label_stack, tmp_folder + "labels_10m.vrt", seperate=False)

    aligned = align_rasters(
        stacked,
        master=reference,
        dst_nodata=None,
    )

    aligned_arr = raster_to_array(aligned)

    gdal.Unlink(stacked)
    gdal.Unlink(aligned)

    if isinstance(aligned_arr, np.ma.MaskedArray):
        aligned_arr.fill_value = 0
        aligned_arr = aligned_arr.filled(0)
    
    if round_label is not None:
        aligned_arr = np.round(aligned_arr, round_label)

    array_to_raster(
        aligned_arr,
        reference=reference,
        out_path=out_path,
        set_nodata=None,
    )

    # clean
    for tmp_label in label_stack:
        try:
            os.remove(tmp_label)
        except:
            pass

    try:
        os.remove(stacked)
    except:
        pass

    return out_path