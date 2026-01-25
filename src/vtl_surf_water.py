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
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from skimage.morphology import local_maxima, local_minima
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter

# --- The data for the surface water vector file (OSM Belgium)
GIS_folder = os.path.join(Path(__file__).resolve().parent.parent, 'data', 'QGIS')
assert os.path.isdir(GIS_folder), f'Cant open GIS folder <{GIS_folder}>'

osm_file = os.path.join(GIS_folder, "gis_osm_waterways_free_1.shp")
output_file = os.path.join(GIS_folder, "waterlopen_15x15km.shp")
assert os.path.isfile(osm_file), f"Can't file file <{osm_file}>"


# %%
def get_national_surf_water_gdf(crs="EPSG:31370", show=False):
    """Return GeoDataFrame with the surface water lines of Belgium from OSM."""
        
    # --- 3. Load OSM (Open Street Map) water layer ---
    water_gdf = gpd.read_file(osm_file)

    # --- 4. Reproject if necessary ---
    if water_gdf.crs != crs:
        water_gdf = water_gdf.to_crs(crs)

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


def get_point_gdf(gr, show=False):
    """
    Return point_gdf, where point is center of grid.
    
    Parameters
    ----------
    gr : Grid object
        grid object holding the fdm network coordinates
    """
    pt_geom = Point((gr.x.mean(), gr.y.mean()))  # point = (x, y) in EPSG:31370
    pt_gdf = gpd.GeoDataFrame(geometry=[pt_geom], crs=gr.crs)
    
    if show:
        pt_gdf.plot(ax=plt.gca(), marker='o', color='black')
    return pt_gdf


def get_tile_gdf(gr, show=False):
    """
    Return tile_gdf, bounded by gr extent.
    
    Parameters
    ----------
    gr : Grid object
        grid object holding the fdm network coordinates
    """

    tile = box(gr.x[0], gr.y[-1], gr.x[-1], gr.y[0])
    tile_gdf = gpd.GeoDataFrame(geometry=[tile], crs=gr.crs)
    
    if show:        
        tile_gdf.boundary.plot(ax=plt.gca(), color='red', linewidth=2)
        
    return tile_gdf
    

def clip_water_to_gr(gr):
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
    crs : str
        CRS to use (default EPSG:31370)
    
    Returns
    -------
    clipped_gdf : GeoDataFrame
        Water features inside the 15x15 km tile
    """    
    tile_gdf = get_tile_gdf(gr)
    
    # tile_gdf is your 15x15 km GeoDataFrame
    xmin, ymin, xmax, ymax = tile_gdf.total_bounds

    # --- 3. Load water layer ---
    water_gdf = gpd.read_file(osm_file)

    # --- 4. Reproject if necessary ---
    if water_gdf.crs != tile_gdf.crs:
        water_gdf = water_gdf.to_crs(gr.crs)
        
    # Quick bbox filter (fast)
    water_bbox = water_gdf.cx[xmin:xmax, ymin:ymax]

    # --- 5. Clip ---
    clipped_gdf = gpd.clip(water_bbox, tile_gdf)

    # --- 6. Save if requested ---
    if output_file is not None:
        clipped_gdf.to_file(output_file)

    print(f"Tile bounds: X={gr.x[0]}-{gr.x[-1]}, Y={gr.y[-1]}-{gr.y[0]}")
    print(f"Original features: {len(water_gdf)}, Clipped features: {len(clipped_gdf)}")

    return clipped_gdf


def line_length_per_gr_cell(gr, polylines_gdf):
    """
    Compute total length of poly_lines per grid cell.
    
    Parameters
    ----------
    gr: Grid object
        grid object holding the grid coordinates
    polylines_gdf : GeoDataFrame
        Clipped waterlines (LineString or MultiLineString), same CRS as polylines_gdf.
    
    Returns
    -------
    grid : GeoDataFrame
        Grid cells with total water length per cell (column 'water_length_m').
    """    

    # --- 1. Grid coordinates are in the grid object ---
    x = gr.x
    y = gr.y
    
    # --- 1. Cell polygons as a comprehension
    polygons = [box(xL, yB, xR, yT)                    
                    for yB, yT in zip(y[1:], y[:-1])
                        for xL, xR in zip(x[:-1], x[1:])]
    
    
    grid_cells_gdf = gpd.GeoDataFrame(geometry=polygons, crs=polylines_gdf.crs)    
    grid_cells_gdf['I'] = np.asarray(gr.NOD[0].flatten(), dtype=int)

    # --- 2. Overlay waterlines with grid cells ---
    intersections = gpd.overlay(polylines_gdf, grid_cells_gdf, how="intersection")

    # --- 3. Compute length (in meters) of each intersection ---
    intersections["length_m"] = intersections.geometry.length

    # --- 4. Spatial join to link intersections to parent grid cell ---
    joined = gpd.sjoin(intersections, grid_cells_gdf, how="left", predicate="intersects")

    # --- 5. Aggregate: total length per cell ---
    lengths_per_cell = joined.groupby("index_right")["length_m"].sum()

    # --- 6. Attach results to grid ---
    grid_cells_gdf["water_length_m"] = grid_cells_gdf.index.map(lengths_per_cell).fillna(0)
    
    return grid_cells_gdf # water length for all cells


def get_water_length_per_cell(gr, show=True):
    """Return Grid object and GHB object specifying surface water within tile of 15x15 km around point in Belgium"""
        
    # --- clip at outer boundary of model grid
    clipped_gdf = clip_water_to_gr(gr)
    
    # --- get water length per cell
    grid_gdf = line_length_per_gr_cell(gr, clipped_gdf)

    if show:
        # --- 5. (Optioneel) visualisatie ---
        # Convert point tuple to GeoDataFrame
        pt_geom = Point((gr.x.mean(), gr.y.mean()))  # point = (x, y) in EPSG:31370
        pt_gdf = gpd.GeoDataFrame(geometry=[pt_geom], crs=gr.crs)

        # --- Plot
        tile_gdf = get_tile_gdf(gr)
        
        ax = tile_gdf.boundary.plot(color="red", linewidth=2, figsize=(8,8))
        
        clipped_gdf.plot(ax=ax, color="blue", linewidth=1)
        
        pt_gdf.plot(ax=ax, color="black", markersize=50)

        ax.set_title("15x15 km modelgebied met oppervlaktewater")
        
        ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'tile_plot.png'))

        # --- Plot
        ax = grid_gdf.plot(column="water_length_m", cmap="Blues", legend=True, figsize=(8,8))
        
        clipped_gdf.plot(ax=ax, color="black", linewidth=0.5)
        pt_gdf.plot(ax=ax, color='red', markersize=50)
        
        ax.set_title("grid tile with density of water features per cell")

        ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'surf_wat_in_cells.png'))
        
    return grid_gdf # Just use column ["water_length"]

def select_15x15km_from_surfwater_belgium(xy=(193919, 194774)):    
    # --- crs of Belgium ---
    crs = "EPSG:31370"
    
    # --- arbitrary location
    x0, y0 = xy # (193919, 194774) # crs 31370
    
    # --- properties to generate a model grid
    wc, w, N = 5, 15000, 100
    z = np.array([0, -30])
    xi = np.logspace(np.log10(wc / 2), np.log10(w / 2), int(N // 2))
    
    x = x0 + np.hstack((-xi[::-1], xi))
    y = y0 + np.hstack((xi[::-1], -xi))
    
    # --- Generate a rectangular Modflow like grid
    gr = Grid(x, y, z, axial=False)
    gr.crs = crs
   
    # --- Alternative (short)
    # grid is geodataframe with total water course length per cell
    grid = get_water_length_per_cell(gr, show=True)
    
    # --- Three GeoDataFrames are used to clip the surface water
    water_gdf = get_national_surf_water_gdf(crs=gr.crs, show=True)    
    point_gdf = get_point_gdf(gr, show=True)
    tile_gdf  = get_tile_gdf(gr, show=True)

    ax = plt.gca()
    
    ax.set_title("Opp. water België met voorbeeld ingreep en 15x15 km modelgebied eromheen")
    ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'ow_Belgie_tile_point.png'))

def climb_to_ridge(x, y, dist=None):
    """Climb from current pixel to ridge.
    
    Parameters
    ----------
    x, y: starting point indices
    dist: raster
        raster with distance to surface water
    """
    x, y = int(x), int(y)  
    path = [(x, y)]

    # eerste staprichting onbekend
    dy_prev, dx_prev = None, None

    ny, nx = dist.shape

    while True:
        current = dist[y, x]

        # 3×3 window
        y0, y1 = max(y-1, 0), min(y+2, ny)
        x0, x1 = max(x-1, 0), min(x+2, nx)
        window = dist[y0:y1, x0:x1]

        dy, dx = np.unravel_index(np.argmax(window), window.shape)
        y_new = y0 + dy
        x_new = x0 + dx

        # nieuwe staprichting
        dy_step = y_new - y
        dx_step = x_new - x

        if dy_prev is not None:
            # vooruitkijken in dezelfde richting
            y_fwd = y + dy_prev
            x_fwd = x + dx_prev

            if (0 <= y_fwd < ny and 0 <= x_fwd < nx and
                dist[y_fwd, x_fwd] <= current):
                    # dwarsmaximum bereikt → graat
                    break

        dy_prev, dx_prev = dy_step, dx_step
        y, x = y_new, x_new
        path.append((x, y))
    return path[-1][0], path[-1][1]


def cleanup_simplify(xy=(193919, 194774), L=15000):
    xm, ym = xy
    x, y, z = [xm - L/2, xm + L/2], [ym - L/2, ym + L/2], [0, -1]
    
    gr = Grid(x, y, z)
    gr.crs = "EPSG:31370"
    
    clip_gdf = clip_water_to_gr(gr)
    # clip_gdf = clip_gdf.explode(ignore_index=True)
    clip_gdf["length_m"] = clip_gdf.length
    min_length = 250 # m
    water_main = clip_gdf[clip_gdf["length_m"] >= min_length].copy()
    water_main = clip_gdf.explode(index_parts=False).reset_index(drop=True)
    ax = water_main.plot(color="lightblue", linewidth=1)
    ax.set_title("Water_main")


    print("Voor:", len(clip_gdf))
    print("Na  :", len(water_main))
    ax = clip_gdf.plot()
    ax.set_title("Original, uncleaned")
    ax = water_main.plot()
    ax.set_title("without small loose pieces")

    water_simpl = water_main.copy()
    water_simpl["geometry"] = water_simpl.geometry.simplify(
            tolerance=100, preserve_topology=True
    )
    ax = water_simpl.plot()
    ax.set_title("Simplified surface water")
    
    pixelsize = 10 #
    xmin, ymin, xmax, ymax = water_simpl.total_bounds
    width = int((xmax - xmin) / pixelsize)
    height = int((ymax - ymin) / pixelsize)
    transform = from_origin(xmin, xmax, pixelsize, pixelsize)
    raster = features.rasterize(
        [(geom, 1) for geom in water_main.geometry], # your geometries with a burn value
        out_shape=(height, width),                          # shape of the raster
        transform=transform,                         # affine transform for your raster
        fill=0,                                 # value for background
        dtype = 'uint8'        
    )
    
    from scipy.ndimage import distance_transform_edt

    dist_to_water = distance_transform_edt(1 - raster) * pixelsize

    def to_xy(pq):
        pq = np.asarray()

    # Get raster shape
    nrows, ncols = dist_to_water.shape
    left = transform.c
    top = transform.f
    right = left + ncols * transform.a
    bottom= top + nrows * transform.e
    extent=[left, right, bottom, top]
    x = left + np.arange(width) * pixelsize
    y = top  - np.arange(height) * pixelsize
    
    fig, ax = plt.subplots(figsize=(10, 8))
    #c=ax.imshow(dist_to_water, cmap='viridis', origin='upper', extent=extent)
    # fig.colorbar(c, ax=ax, label='Distance to nearest water (m)')
    ax.set_title('Distance to Surface Water')

    obs_coordinates = np.array([
       [194591., 196868.],
       [193362., 197692.],
       [193214., 195058.],
       [193470., 195612.],
       [192660., 195882.],
       [192876., 194018.],
       [192619., 194275.],
       [192430., 193816.],
       [192890., 193478.]])

    contours = ax.contour(x, y, dist_to_water, levels=20, colors='black', linewidths=0.5, origin='upper')
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    
    water_simpl.plot(ax=ax)
    ax.set_title("Voorbeeld opp.water met afstandskaart en denkbeeldige waarnemingsputten")
    # ax.plot(*obs_coordinates.T, 'ro', label="observation points")
    
    xi, yi = ~transform * (obs_coordinates[:, 0], obs_coordinates[:, 1])
    xi, yi = np.asarray(xi, dtype=int), np.asarray(yi, dtype=int)
    
    obs_ridge = []
    bs = []
    for x, y in obs_coordinates:
        ix, iy = ~transform * (x, y)
        ix_, iy_ = climb_to_ridge(ix, iy, dist_to_water)
        x_, y_ = transform * (float(ix_), float(iy_))
        bs.append(dist_to_water[iy_, ix_])
        obs_ridge.append((x_, y_))
    obs_ridge = np.array(obs_ridge, dtype=float)    
    xx = np.column_stack([obs_coordinates[:, 0], obs_ridge[:, 0]]).T
    yy = np.column_stack([obs_coordinates[:, 1], obs_ridge[:, 1]]).T
    ax.plot(xx, yy, '.-', label='climb')
    print("xDupuit = ", np.round(np.sqrt(((obs_coordinates - obs_ridge) ** 2).sum(axis=1))))
    print("bDupuit = ", np.round(np.array(bs)))
    
        

# %%
if __name__ == '__main__':
    if False:
        select_15x15km_from_surfwater_belgium(xy=(193919, 194774))
    if True:
        cleanup_simplify(xy=(193919, 194774), L=15000)
    plt.show()
    
# %%
