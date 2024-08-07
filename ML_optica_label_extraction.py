import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
import matplotlib.pyplot as plt
from rasterio.features import rasterize

def extract_data(boundary_shp_path, crop_shp_path):
    # Load the boundary and crop shapefiles
    boundary = gpd.read_file(boundary_shp_path)
    crop_shapes = gpd.read_file(crop_shp_path)

    # File discovery for Sentinel-2 data
    s2_files = sorted([f for f in os.listdir('.') if f.startswith('Sentinel-2_')])

    # Ensure CRS is consistent across all geospatial data
    with rasterio.open(s2_files[0]) as src:
        if boundary.crs != src.crs:
            boundary = boundary.to_crs(src.crs)
        if crop_shapes.crs != src.crs:
            crop_shapes = crop_shapes.to_crs(src.crs)

        # Masking operation to create the boundary of interest
        out_image, out_transform = mask(src, boundary.geometry, crop=True, all_touched=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
        data_array = np.full((len(s2_files), out_image.shape[0], out_image.shape[1], out_image.shape[2]), np.nan)

    # Processing each time step
    for i, file in enumerate(s2_files):
        with rasterio.open(file) as src:
            data, _ = mask(src, boundary.geometry, crop=True, all_touched=True)
            data_array[i] = data.squeeze()

    # Rasterizing crop codes
    label_array = rasterize(
        [(row.geometry, int(row['Crop_code'])) for idx, row in crop_shapes.iterrows()],
        out_shape=data_array.shape[2:],
        fill=np.nan,
        transform=out_transform,
        dtype=np.float32
    )

    print(label_array.shape)
    print(data_array.shape)
    np.save("output_data.npy", data_array)
    np.save("output_labels.npy", label_array)

    # Saving GeoTIFFs for labels
    with rasterio.open("output_labels.tif", "w", **out_meta) as dest:
        dest.write(label_array, 1)

    with rasterio.open("output_data.tif", "w", **out_meta) as dest:
        for i in range(data_array.shape[1]):
            dest.write(data_array[0, i], i+1)

    # Plotting the first dataset and the label array
    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(data_array[0, 0], cmap='viridis')
    plt.colorbar()
    plt.title('Sample Band 1 Data')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(label_array, cmap='viridis')
    plt.colorbar()
    plt.title('Crop Type Labels')
    plt.show()

boundary_shp_path = "shapefile/gialia_test_boundary2_07-29.shp"
crop_shp_path = "shapefile/gialia_test2_07-29.shp"
extract_data(boundary_shp_path, crop_shp_path)
