# %% [markdown]
# # Selecteer oppervlaktewater België in een vierkant rond punt.
#

# %%
# Standard library
import math
import os
from pathlib import Path

# Third-party
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box, Point # noqa: I001

# %%
def clip_water_15km(point, osm_file, output_file=None, tile_size=15000, cell_size=1000, target_crs="EPSG:31370"):
    """
    Clip water features within a 15x15 km tile centered around a point.
    
    Parameters
    ----------
    point : tuple
        Coordinates (x, y) in target CRS (EPSG:31370)
    osm_file : str
        Path to the input water shapefile or GeoPackage
    output_file : str, optional
        If provided, the clipped GeoDataFrame will be written to this file
    tile_size : float
        Total tile size in meters (default 15000)
    cell_size : float
        Size of the internal 1x1 km grid (for centering the point, default 1000)
    target_crs : str
        CRS to use (default EPSG:31370)
    
    Returns
    -------
    clipped : GeoDataFrame
        Water features inside the 15x15 km tile
    """
    
    x, y = point

    # --- 1. Compute center of the middle 1x1 km cell ---
    x_center = round(x / cell_size) * cell_size
    y_center = round(y / cell_size) * cell_size

    # --- 2. Compute the 15x15 km tile boundaries ---
    x0 = x_center - (tile_size / 2)
    y0 = y_center - (tile_size / 2)
    x1 = x0 + tile_size
    y1 = y0 + tile_size

    # Snap to whole km lines
    x0 = math.floor(x0 / cell_size) * cell_size
    y0 = math.floor(y0 / cell_size) * cell_size
    x1 = math.ceil(x1 / cell_size) * cell_size
    y1 = math.ceil(y1 / cell_size) * cell_size

    tile = box(x0, y0, x1, y1)
    tile_gdf = gpd.GeoDataFrame(geometry=[tile], crs=target_crs)
    
    # tile_gdf is your 15x15 km GeoDataFrame
    xmin, ymin, xmax, ymax = tile_gdf.total_bounds

    # --- 3. Load water layer ---
    water = gpd.read_file(osm_file)

    # --- 4. Reproject if necessary ---
    if water.crs != tile_gdf.crs:
        water = water.to_crs(tile_gdf.crs)

    # Quick bbox filter (fast)
    water_bbox = water.cx[xmin:xmax, ymin:ymax]

    # --- 5. Clip ---
    clipped = gpd.clip(water_bbox, tile_gdf)

    # --- 6. Save if requested ---
    if output_file is not None:
        clipped.to_file(output_file)

    print(f"Tile bounds: X={x0}-{x1}, Y={y0}-{y1}")
    print(f"Original features: {len(water)}, Clipped features: {len(clipped)}")

    return clipped, tile_gdf


def water_length_per_cell(clipped, tile_gdf, cell_size=1000):
    """
    Compute total waterway length per grid cell inside a tile.
    
    Parameters
    ----------
    clipped : GeoDataFrame
        Clipped waterlines (LineString or MultiLineString), same CRS as tile_gdf.
    tile_gdf : GeoDataFrame
        Single-tile GeoDataFrame defining the 15x15 km area (EPSG:31370).
    cell_size : float
        Size of grid cell in meters (default = 1000).
    
    Returns
    -------
    grid : GeoDataFrame
        Grid cells with total water length per cell (column 'water_length_m').
    """

    # --- 1. Create regular grid covering the tile ---
    xmin, ymin, xmax, ymax = tile_gdf.total_bounds
    cols = np.arange(xmin, xmax, cell_size)
    rows = np.arange(ymin, ymax, cell_size)
    polygons = [box(x, y, x + cell_size, y + cell_size) for x in cols for y in rows]
    grid = gpd.GeoDataFrame(geometry=polygons, crs=tile_gdf.crs)

    # --- 2. Overlay waterlines with grid cells ---
    intersections = gpd.overlay(clipped, grid, how="intersection")

    # --- 3. Compute length (in meters) of each intersection ---
    intersections["length_m"] = intersections.geometry.length

    # --- 4. Spatial join to link intersections to parent grid cell ---
    joined = gpd.sjoin(intersections, grid, how="left", predicate="intersects")

    # --- 5. Aggregate: total length per cell ---
    lengths_per_cell = joined.groupby("index_right")["length_m"].sum()

    # --- 6. Attach results to grid ---
    grid["water_length_m"] = grid.index.map(lengths_per_cell).fillna(0)

    return grid
# %%

GIS_folder = os.path.join(Path('__file__').resolve().parent, 'data', 'QGIS')
assert os.path.isdir(GIS_folder), f'Cant open GIS folder <{GIS_folder}>'

osm_file = os.path.join(GIS_folder, "gis_osm_waterways_free_1.shp")
output_file = os.path.join(GIS_folder, "waterlopen_15x15km.shp")

assert os.path.isfile(osm_file), f"Can't file file <{osm_file}>"

target_crs = "EPSG:31370"

point = (193919, 194774)

clipped, tile_gdf = clip_water_15km(point, osm_file, output_file=output_file, tile_size=15000, cell_size=1000, target_crs=target_crs)

# --- 5. (Optioneel) visualisatie ---
# Convert point tuple to GeoDataFrame
pt_geom = Point(point)  # point = (x, y) in EPSG:31370
pt_gdf = gpd.GeoDataFrame(geometry=[pt_geom], crs=tile_gdf.crs)

# Plot
ax = tile_gdf.boundary.plot(color="red", linewidth=2, figsize=(8,8))
clipped.plot(ax=ax, color="blue", linewidth=1)
pt_gdf.plot(ax=ax, color="black", markersize=50)

ax.set_title("15x15 km Tile with Clipped Water Features")


# %%
# Assuming you already have:
# clipped : GeoDataFrame (clipped waterways, EPSG:31370)
# tile_gdf : GeoDataFrame (the 15x15 km tile polygon)

grid = water_length_per_cell(clipped, tile_gdf, cell_size=250)

print(grid.head())

# --- Visualize ---
ax = grid.plot(column="water_length_m", cmap="Blues", legend=True, figsize=(8,8))
clipped.plot(ax=ax, color="black", linewidth=0.5)

plt.show()
