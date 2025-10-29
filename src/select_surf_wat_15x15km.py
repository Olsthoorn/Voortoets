# %% [markdown]
# # Selecteer oppervlaktewater België in een vierkant rond punt.
#

# %%
# Standard library
import os
from pathlib import Path
import numpy as np

# Third-party
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box, Point # noqa: I001

from fdm.src.mfgrid import Grid

# --- The data for the surface water vector file (OSM Belgium)
GIS_folder = os.path.join(Path(__file__).resolve().parent.parent, 'data', 'QGIS')
assert os.path.isdir(GIS_folder), f'Cant open GIS folder <{GIS_folder}>'

osm_file = os.path.join(GIS_folder, "gis_osm_waterways_free_1.shp")
output_file = os.path.join(GIS_folder, "waterlopen_15x15km.shp")
assert os.path.isfile(osm_file), f"Can't file file <{osm_file}>"


# %%
def get_water_gdf(target_crs="EPSG:31370", show=True):
        
    # --- 3. Load OSM water layer ---
    water_gdf = gpd.read_file(osm_file)

    # --- 4. Reproject if necessary ---
    if water_gdf.crs != target_crs:
        water_gdf = water_gdf.to_crs(target_crs)

    if show:
        # ax = water.boundary.plot(color="red", linewidth=1, figsize=(8,8))
        water_gdf.plot(ax=None, color='blue', linewidth=0.25, figsize=(12, 10))
        ax = plt.gca()
        
        ax.set(title="Oppervlaktewater België (Open Street Map)",
            xlabel='x [crs=31370]',
            ylabel='y [crs=31370]')        
        ax.grid(True)
    
        ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'oppwat_Belgie.png'))
        
    return water_gdf


def get_point_gdf(gr, target_crs="EPSG:31370", show=True):
    """
    Return point_gdf, where point is center of grid.
    
    Parameters
    ----------
    gr : Grid object
        grid object holding the fdm network coordinates
    target_crs : str
        CRS to use (default EPSG:31370)    
    """
    pt_geom = Point((gr.x.mean(), gr.y.mean()))  # point = (x, y) in EPSG:31370
    pt_gdf = gpd.GeoDataFrame(geometry=[pt_geom], crs=target_crs)
    
    if show:
        pt_gdf.plot(ax=plt.gca(), marker='o', color='black')
    return pt_gdf


def get_tile_gdf(gr, target_crs="EPSG:31370", show=True):
    """
    Return tile_gdf, bounded by gr extent.
    
    Parameters
    ----------
    gr : Grid object
        grid object holding the fdm network coordinates
    target_crs : str
        CRS to use (default EPSG:31370)    
    """

    tile = box(gr.x[0], gr.y[-1], gr.x[-1], gr.y[0])
    tile_gdf = gpd.GeoDataFrame(geometry=[tile], crs=target_crs)
    
    if show:        
        tile_gdf.boundary.plot(ax=plt.gca(), color='red', linewidth=2)
        
    return tile_gdf
    

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
        If provided, the clipped_gdf GeoDataFrame will be written to this file
    target_crs : str
        CRS to use (default EPSG:31370)
    
    Returns
    -------
    clipped_gdf : GeoDataFrame
        Water features inside the 15x15 km tile
    """    
    tile_gdf = get_tile_gdf(gr, target_crs=target_crs)
    
    # tile_gdf is your 15x15 km GeoDataFrame
    xmin, ymin, xmax, ymax = tile_gdf.total_bounds

    # --- 3. Load water layer ---
    water_gdf = gpd.read_file(osm_file)

    # --- 4. Reproject if necessary ---
    if water_gdf.crs != tile_gdf.crs:
        water_gdf = water_gdf.to_crs(tile_gdf.crs)
        
    # Quick bbox filter (fast)
    water_bbox = water_gdf.cx[xmin:xmax, ymin:ymax]

    # --- 5. Clip ---
    clipped_gdf = gpd.clip(water_bbox, tile_gdf)

    # --- 6. Save if requested ---
    if output_file is not None:
        clipped_gdf.to_file(output_file)

    print(f"Tile bounds: X={gr.x[0]}-{gr.x[-1]}, Y={gr.y[-1]}-{gr.y[0]}")
    print(f"Original features: {len(water_gdf)}, Clipped features: {len(clipped_gdf)}")

    return clipped_gdf, tile_gdf


def cells_with_points(gr, points):
    """
    Get cell Id of points.
    
    Parameters
    ----------
    gr: Grid object
        grid object holding the grid coordinates
    points_gdf : GeoDataFrame
        GeoDataFrame with geometry Points.
    
    Returns
    -------
    grid : GeoDataFrame
        Cell Id in which the points lie.
    """
    # --- 1. Grid coordinates are in the grid object ---
    coords = []
    for point in points:
        coords.append([[point['geometry'].x, point['geometry'].y]])
    coords = np.vstack(coords)
    
    xw, yw = coords.T
    
    zw = []
    for point in points:
        if 'depth_aquifer' == 'A0800':
            zw.append(gr.zm[0])
        else:
            zw.append(gr.zm[0])
    zw = np.array(zw)
    
    I = gr.Iglob_from_xyz(np.vstack((xw, yw, zw)).T)

    return I


def cells_in_pgons(gr, pgons):
    """
    Get cell Ids of points in union of polygons.
    
    Parameters
    ----------
    gr: Grid object
        grid object holding the grid coordinates
    points_gdf : GeoDataFrame
        GeoDataFrame with geometry Points.
    
    Returns
    -------
    grid : GeoDataFrame
        Cell Id in which the points lie.
    """
    # --- 1. Grid coordinates are in the grid object ---
    mask = np.zeros((gr.ny, gr.nx), dtype=bool)
    
    for pgon in pgons:
        mask = np.logical_or(mask, gr.inpoly(pgon))
    
    Id = gr.NOD[0][mask].flatten
    return Id


def line_length_per_cell(gr, polylines_gdf):
    """
    Compute total length of poly_lines per grid cell.
    
    Parameters
    ----------
    gr: Grid object
        grid object holding the grid coordinates
    polylines_gdf : GeoDataFrame
        Clipped waterlines (LineString or MultiLineString), same CRS as tile_gdf.
    
    Returns
    -------
    grid : GeoDataFrame
        Grid cells with total water length per cell (column 'water_length_m').
    """    
    for pline in plines:

    # --- 1. Grid coordinates are in the grid object ---
    x = gr.x
    y = gr.y
    
    # --- 1. Cell polygons
    polygons = [box(xL, yB, xR, yT)
                for xL, xR in zip(x[:-1], x[1:])
                for yB, yT in zip(y[1:], y[:-1])]
    
    grid_cells = gpd.GeoDataFrame(geometry=polygons, crs=tile_gdf.crs)    
    grid_cells['I'] = np.asarray(gr.NOD[0].flatten(), dtype=int)

    # --- 2. Overlay waterlines with grid cells ---
    intersections = gpd.overlay(polylines_gdf, grid_cells, how="intersection")

    # --- 3. Compute length (in meters) of each intersection ---
    intersections["length_m"] = intersections.geometry.length

    # --- 4. Spatial join to link intersections to parent grid cell ---
    joined = gpd.sjoin(intersections, grid_cells, how="left", predicate="intersects")

    # --- 5. Aggregate: total length per cell ---
    lengths_per_cell = joined.groupby("index_right")["length_m"].sum()

    # --- 6. Attach results to grid ---
    grid_cells["water_length_m"] = grid_cells.index.map(lengths_per_cell).fillna(0)
    
    return grid_cells


# %%
def get_water_length_per_cell(gr, show=True):
    """Return Grid object and GHB object specifying surface water within tile of 15x15 km around point in Belgium"""
        
    # --- clip at outer boundary of model grid
    clipped_gdf, tile_gdf = clip_water_15km(gr, target_crs="EPSG:31370")
    
    # --- get water length per cell
    grid_gdf = water_length_per_cell(gr, clipped_gdf, tile_gdf)

    if show:
        # --- 5. (Optioneel) visualisatie ---
        # Convert point tuple to GeoDataFrame
        pt_geom = Point((gr.x.mean(), gr.y.mean()))  # point = (x, y) in EPSG:31370
        pt_gdf = gpd.GeoDataFrame(geometry=[pt_geom], crs=tile_gdf.crs)

        # --- Plot
        ax = tile_gdf.boundary.plot(color="red", linewidth=2, figsize=(8,8))
        
        clipped_gdf.plot(ax=ax, color="blue", linewidth=1)
        
        pt_gdf.plot(ax=ax, color="black", markersize=50)

        ax.set_title("15x15 km modelgebied met oppervlaktewater")
        
        ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'tile_plot.png'))

        # --- Plot
        ax = grid_gdf.plot(column="water_length_m", cmap="Blues", legend=True, figsize=(8,8))
        
        clipped_gdf.plot(ax=ax, color="black", linewidth=0.5)
        
        ax.set_title("15x15 km Tile with density of water features per cell")

        ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'surf_wat_in_cells.png'))
        
    return grid_gdf

# %%
if __name__ == '__main__':
    x0, y0 = (193919, 194774) # crs 31370
    
    wc, w, N = 5, 15000, 100
    z = np.array([0, -30])
    xi = np.logspace(np.log10(wc / 2), np.log10(w / 2), int(N // 2))
    
    x = x0 + np.hstack((-xi[::-1], xi))
    y = y0 + np.hstack((xi[::-1], -xi))
    
    gr = Grid(x, y, z, axial=False)
   
    # --- Alternative (short)
    # grid is geodataframe with total water course length per cell
    grid = get_water_length_per_cell(gr, show=True)
    
    target_crs = "EPSG:31370"
    water_gdf = get_water_gdf(target_crs=target_crs, show=True)    
    point_gdf = get_point_gdf(gr, target_crs=target_crs, show=True)
    tile_gdf  = get_tile_gdf(gr, target_crs=target_crs, show=True)

    ax = plt.gca()
    
    ax.set_title("Opp. water België met voorbeeld ingreep en 15x15 km modelgebied eromheen")
    ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'ow_Belgie_tile_point.png'))

    plt.show()
    

