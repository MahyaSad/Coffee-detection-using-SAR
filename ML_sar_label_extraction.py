# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:53:29 2024

@author: ghaziza1
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from shapely.geometry import Polygon, LinearRing

def extract_data(boundary_shp_path, crop_shp_path):
    # Load the boundary and crop shapefiles
    boundary = gpd.read_file(boundary_shp_path)
    crop_shapes = gpd.read_file(crop_shp_path)

    # File discovery for SAR data
    vh_files = sorted([f for f in os.listdir('.') if 'VH' in f])
    vv_files = sorted([f for f in os.listdir('.') if 'VV' in f])
    ia_files = sorted([f for f in os.listdir('.') if 'map' in f])  # Adjusted tag for IA files

    # Ensure CRS is consistent across all geospatial data
    with rasterio.open(vh_files[0]) as src:
        if boundary.crs != src.crs:
            boundary = boundary.to_crs(src.crs)
        if crop_shapes.crs != src.crs:
            crop_shapes = crop_shapes.to_crs(src.crs)

        # Masking operation to create the boundary of interest
        out_image, out_transform = mask(src, boundary.geometry, crop=True, all_touched=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
        data_array = np.full((len(vh_files), 5, out_image.shape[1], out_image.shape[2]), np.nan)

    # Processing each time step
    for i, (vh, vv, ia) in enumerate(zip(vh_files, vv_files, ia_files)):
        with rasterio.open(vh) as vh_src, rasterio.open(vv) as vv_src, rasterio.open(ia) as ia_src:
            vh_data, _ = mask(vh_src, boundary.geometry, crop=True, all_touched=True)
            vv_data, _ = mask(vv_src, boundary.geometry, crop=True, all_touched=True)
            ia_data, _ = mask(ia_src, boundary.geometry, crop=True, all_touched=True)
            data_array[i, 0] = vv_data.squeeze()
            data_array[i, 1] = vh_data.squeeze()
            data_array[i, 2] = ia_data.squeeze()

            # Converting to linear scale
            VH_linear = 10 ** (vh_data / 10)
            VV_linear = 10 ** (vv_data / 10)

            # Calculate RVI and VH/VV ratio
            rvi = np.where((VH_linear + VV_linear) != 0, (4 * VH_linear) / (VH_linear + VV_linear), 0)
            vh_vv_ratio = np.where(VV_linear != 0, VH_linear / VV_linear, np.nan)

            # Store additional layers if necessary
            data_array[i, 3] = vh_vv_ratio.squeeze() 
            data_array[i, 4] = rvi.squeeze()

    # Rasterizing crop codes
    # Rasterizing crop codes
    label_array = rasterize(
        [(row.geometry, int(row['Crop_code'])) for idx, row in crop_shapes.iterrows()],
        out_shape=data_array.shape[2:],
        fill=np.nan,
        transform=out_transform,
        dtype=np.float32
    )


    
    print(label_array.shape)
    np.save("output_data.npy", data_array)
    np.save("output_labels.npy", label_array)

    # Saving GeoTIFFs for labels
    with rasterio.open("output_labels2.tif", "w", **out_meta) as dest:
        dest.write(label_array, 1)
        
    with rasterio.open("output_data2.tif", "w", **out_meta) as dest:
        dest.write(data_array[0,0,:,:], 1)

    # Plotting the first VH dataset and the label array
    plt.figure(figsize=(10, 10),dpi=300)
    plt.imshow(data_array[0, 0], cmap='viridis')
    plt.colorbar()
    plt.title('Sample VH Data')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(label_array, cmap='viridis')
    plt.colorbar()
    plt.title('Crop Type Labels')
    plt.show()

# Usage of the function
boundary_shp_path = "shapefile/gialia_test_boundary2_07-29.shp"
crop_shp_path = "shapefile/gialia_test2_07-29.shp"
extract_data(boundary_shp_path, crop_shp_path)
