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

from fdm.src.fdm3t import dtypeQ, dtypeH, dtypeGHB
from fdm.src.mfgrid import Grid

# --- The data for the surface water vector file (OSM Belgium)
GIS_folder = os.path.join(Path(__file__).resolve().parent.parent, 'data', 'QGIS')
assert os.path.isdir(GIS_folder), f'Cant open GIS folder <{GIS_folder}>'

osm_file = os.path.join(GIS_folder, "gis_osm_waterways_free_1.shp")
output_file = os.path.join(GIS_folder, "waterlopen_15x15km.shp")
assert os.path.isfile(osm_file), f"Can't file file <{osm_file}>"


# %%
def clip_water_15km(gr, target_crs="EPSG:31370"):
    """
    Clip water features within a 15x15 km tile centered around a point.
    
    Parameters
    ----------
    gr : Grid object
        grid object holding the fdm network coordinates
    osm_file : str
        Path to the input water shapefile or GeoPackage
    output_file : str, optional
        If provided, the clipped GeoDataFrame will be written to this file
    target_crs : str
        CRS to use (default EPSG:31370)
    
    Returns
    -------
    clipped : GeoDataFrame
        Water features inside the 15x15 km tile
    """

    tile = box(gr.x[0], gr.y[-1], gr.x[-1], gr.y[0])
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

    print(f"Tile bounds: X={gr.x[0]}-{gr.x[-1]}, Y={gr.y[-1]}-{gr.y[0]}")
    print(f"Original features: {len(water)}, Clipped features: {len(clipped)}")

    return clipped, tile_gdf


def water_length_per_cell(gr, clipped, tile_gdf):
    """
    Compute total waterway length per grid cell inside a tile.
    
    Parameters
    ----------
    gr: Grid object
        grid object holding the grid coordinates
    clipped : GeoDataFrame
        Clipped waterlines (LineString or MultiLineString), same CRS as tile_gdf.
    tile_gdf : GeoDataFrame
        Single-tile GeoDataFrame defining the 15x15 km area (EPSG:31370).
    
    Returns
    -------
    grid : GeoDataFrame
        Grid cells with total water length per cell (column 'water_length_m').
    """

    # --- 1. Grid coordinates are in the grid object ---
    x = gr.x
    y = gr.y
    
    # --- 1. Cell polygons
    polygons = [box(xL, yB, xR, yT)
                for xL, xR in zip(x[:-1], x[1:])
                for yB, yT in zip(y[1:], y[:-1])]
    
    grid = gpd.GeoDataFrame(geometry=polygons, crs=tile_gdf.crs)    
    grid['I'] = np.asarray(gr.NOD[0].flatten(), dtype=int)

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
def get_water_length_per_cell(gr, show=True):
    """Return Grid object and GHB object specifying surface water within tile of 15x15 km around point in Belgium"""
        
    # --- clip at outer boundary of model grid
    clipped, tile_gdf = clip_water_15km(gr, target_crs="EPSG:31370")
    
    # --- get water length per cell
    grid = water_length_per_cell(gr, clipped, tile_gdf)

    if show:
        # --- 5. (Optioneel) visualisatie ---
        # Convert point tuple to GeoDataFrame
        pt_geom = Point((gr.x.mean(), gr.y.mean()))  # point = (x, y) in EPSG:31370
        pt_gdf = gpd.GeoDataFrame(geometry=[pt_geom], crs=tile_gdf.crs)

        # --- Plot
        ax = tile_gdf.boundary.plot(color="red", linewidth=2, figsize=(8,8))
        clipped.plot(ax=ax, color="blue", linewidth=1)
        pt_gdf.plot(ax=ax, color="black", markersize=50)

        ax.set_title("15x15 km Tile with Clipped Water Features")

        # --- Plot
        ax = grid.plot(column="water_length_m", cmap="Blues", legend=True, figsize=(8,8))
        clipped.plot(ax=ax, color="black", linewidth=0.5)

        plt.show()
        
    return grid

# %%
if __name__ == '__main__':
    x0, y0 = (193919, 194774) # crs 31370
    
    wc, w, N = 5, 15000, 100
    z = np.array([0, -30])
    xi = np.logspace(np.log10(wc / 2), np.log10(w), int(N % 2))
    
    x = x0 + np.hstack((-xi[::-1], xi))
    y = y0 + np.hstack((xi[::-1], -xi))
    
    gr = Grid(x, y, z, axial=False)
   
    # clipped, tile_gdf = clip_water_15km(gr, target_crs="EPSG:31370")
    # grid = water_length_per_cell(gr, clipped, tile_gdf)
    
    # --- Alternative (short)
    # grid is geodataframe with total water course length per cell
    grid = get_water_length_per_cell(gr, show=True)

