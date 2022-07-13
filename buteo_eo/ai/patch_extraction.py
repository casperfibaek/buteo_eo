"""
This module generates patches/tiles from a raster.

TODO:
    - Improve documentation
    - Explain options
"""

# standard
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
    merge_output=True,
    border_check=True,
    force_align=False,
    aoi_mask=None,
    aoi_mask_tolerance=0.0,
    aoi_mask_layerid=0,
    aoi_mask_output=False,
    aoi_mask_apply=False,
    binary_mask=None,
    binary_mask_apply=None,
    binary_mask_fill_value=0,
    prefix="",
    postfix="",
    thread_id="",
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

    for idx, raster in enumerate(raster_list):
        name = os.path.splitext(os.path.basename(raster))[0]
        list_extracted = []
        list_masks = []

        for fid in fids:
            feature = zones_layer.GetFeature(fid)
            geom = feature.GetGeometryRef()
            fid_path = f"/vsimem/{thread_id}fid_mem_{uuid4().int}_{str(fid)}.shp"
            fid_ds = mem_driver.CreateDataSource(fid_path)
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

            uuid_valid = str(uuid4().int)
            valid_path = f"/vsimem/{thread_id}validmask_{str(fid)}_{uuid_valid}_{postfix}.tif"

            if binary_mask is not None:
                extent = clip_raster(
                    binary_mask,
                    clip_geom=fid_path,
                    adjust_bbox=True,
                    all_touch=False,
                    to_extent=True,
                )

                tmp_rasterized_vector = rasterize_vector(
                    fid_path,
                    binary_mask,
                    extent=extent,
                )

                resample_raster(
                    tmp_rasterized_vector,
                    target_size=raster,
                    resample_alg="nearest",
                    out_path=valid_path,
                    postfix="",
                )

                gdal.Unlink(tmp_rasterized_vector)
            else:
                uuid_extent = str(uuid4().int)
                extent = clip_raster(
                    raster,
                    clip_geom=fid_path,
                    out_path=f"/vsimem/{thread_id}tmp_extent_raster_{uuid_extent}_{str(idx)}.tif",
                    adjust_bbox=True,
                    all_touch=False,
                    to_extent=True,
                )

                rasterize_vector(
                    fid_path,
                    (raster_metadata["pixel_width"], raster_metadata["pixel_height"]),
                    out_path=valid_path,
                    extent=extent,
                )

            gdal.Unlink(extent)
            valid_arr = raster_to_array(valid_path)

            uuid = str(uuid4().int)

            raster_clip_path = f"/vsimem/{thread_id}raster_{uuid}_{str(idx)}_clipped.tif"

            try:
                clip_raster(
                    raster,
                    clip_geom=valid_path,
                    out_path=raster_clip_path,
                    all_touch=False,
                    adjust_bbox=False,
                )
            except Exception as error_message:
                print(f"Warning: {raster} did not intersect geom with fid: {fid}.")
                print(error_message)

                gdal.Unlink(fid_path)
                continue

            arr = raster_to_array(raster_clip_path)

            gdal.Unlink(raster_clip_path)

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

            if aoi_mask_apply:
                arr = np.ma.masked_array(arr, mask=valid_masked, fill_value=0)
            
            if merge_output:
                list_extracted.append(arr)
                list_masks.append(valid_masked)
            else:
                out_path = f"{outdir}{prefix}{str(fid)}_{name}{postfix}.npy"
                np.save(out_path, arr.filled(binary_mask_fill_value))

                outputs.append(out_path)

                if aoi_mask_output:
                    np.save(
                        f"{outdir}{prefix}{str(fid)}_mask_{name}{postfix}.npy",
                        valid_masked.filled(binary_mask_fill_value),
                    )
                
            if fid not in processed_fids:
                processed_fids.append(fid)

            processed += 1

            if verbose:
                progress(processed, len(fids) * len(raster_list), "processing fids")

            if not merge_output:
                gdal.Unlink(fid_path)

            gdal.Unlink(valid_path)

        if merge_output:

            if aoi_mask is not None and aoi_mask_apply:
                out_arr = np.ma.concatenate(list_extracted)
            else:
                out_arr = np.ma.concatenate(list_extracted).filled(binary_mask_fill_value)


            out_path = f"{outdir}{prefix}{name}{postfix}.npy"

            if binary_mask_apply is None:
                binary_mask_apply = np.ones(out_arr.shape[0], dtype="bool")
            else:
                binary_mask_apply = binary_mask_apply

            np.save(out_path, out_arr[binary_mask_apply])
            outputs.append(out_path)

            if aoi_mask_output:
                np.save(
                    f"{outdir}{prefix}mask_{name}{postfix}.npy",
                    np.ma.concatenate(list_masks).filled(binary_mask_fill_value)[binary_mask_apply],
                )
    if aoi_mask is not None:
        gdal.Unlink(aoi_mask)

    if input_was_single_raster:
        return outputs[0]

    return outputs


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