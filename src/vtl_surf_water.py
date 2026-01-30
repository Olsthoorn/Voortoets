# %% [markdown]
# # Selecteer oppervlaktewater België in een vierkant vak
#
# Bepaal de drainagweerstand in een realistisch gebied met random oppervlaktewater.
# Dit gaat door eerst een raster te maken met de afstand tot het oppervlaktewater
# als gemodelleerde eigenschap. Hiermee kan voor willekeurige punten worden bepaald
# wat de halve breedte is van de kortste lijn er doorheen, waar deze waterscheiding
# ligt en wat de afstand van het waarnemingspunt is tot de waterscheiding. De
# aftand van het oppervlaktewater tot de waterscheiding samen met de afstand van
# de waarnemingsput tot de waterscheiding zijn de afstanden nodig om het verloop
# van de grondwaterstand langs de doorsnede te berekenen en om daaruit de drainage
# weerstand te bepalen (vergt meerdere putten) en de drainageweerstand van het
# gebied te bepalen. Met de drainageweerstand kan de invloed van een ingreep
# in het grondwatersysteem eenvoudiger worden berekend met formules van
# De Glee en Hantush voor semi-spanningswater. Dit is al gebruikelijk voor
# polders met een dicht stelsel van sloten. Maar kan zo ook voor gebieden
# zonder zo'n dicht stelsel. De vraag is onder welke voorwaarden deze aanpak
# acceptabel is.
#
# De vraag is voorgelegd door David Simpson van AGT in het kader van het onderzoek
# naar verbetering van de berekeningen in de website van de Voortoets van het
# Agentschap Bos en Natuur van de Vlaamse Overheid.
#
# Zie ook file hantush_compromise.py

# TO 20260130
# 
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
from scipy.ndimage import maximum_filter, distance_transform_edt

# --- Get folder for saving images
cwd = os.getcwd()
parts = Path(cwd).parts
assert '2022-AGT' in parts, "2022-AGT must be in path for correct saving: {home}"
images = os.path.join(*parts[:parts.index("2022-AGT") + 1], "Coding", "images")

# --- The data for the surface water vector file (OSM Belgium)
GIS_folder = os.path.join(Path(__file__).resolve().parent.parent, 'data', 'QGIS')
assert os.path.isdir(GIS_folder), f'Cant open GIS folder <{GIS_folder}>'

# --- The the OSM file name and assure its existance
osm_file = os.path.join(GIS_folder, "gis_osm_waterways_free_1.shp")
output_file = os.path.join(GIS_folder, "waterlopen_15x15km.shp")
assert os.path.isfile(osm_file), f"Can't file file <{osm_file}>"

class Raster:
    def __init__(self, array, transform=None, crs=None, nodata=None):
        self.array = array
        self.transform = transform
        self.crs = crs
        self.nodata = nodata
        self.shape = array.shape
        
        if transform is not None:
            self.dx = transform.a
            self.dy = -transform.e

            if not np.isclose(self.dx, self.dy):
                raise ValueError("Non-square pixels not supported")

            self.pixelsize = self.dx
        else:
            self.dx = self.dy = self.pixelsize = None


    def copy(self):
        return Raster(self.array.copy(), transform=self.transform, crs=self.crs, nodata=self.nodata)
    
    def write(self, filename, dtype=None):        
        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
            height=self.shape[0],
            width=self.shape[1],
            count=1,
            dtype=dtype or self.array.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
        ) as dst:
            dst.write(self.array, 1)

    
    
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
        ax = water_gdf.plot(ax=None, color='blue', linewidth=0.25, figsize=(12, 10))
        # ax = plt.gca()
        
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
    # --- Generate a Point object of the center of the grid
    pt_geom = Point((gr.x.mean(), gr.y.mean()))  # point = (x, y) in EPSG:31370
    
    # --- Geneate a geoDataFrame with only one Point
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
    Clip Belgian surface water to within the boundary of the grid gr.
    
    Reads OSM (open street map) of Belgium, with the surface water lines.
    
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
    
    # --- tile_gdf is GeoDataFrame with one record, the rectangle around the grid gr
    tile_gdf = get_tile_gdf(gr)
    
    # --- extent of the tile_gdf
    xmin, ymin, xmax, ymax = tile_gdf.total_bounds

    # --- 3. Load water layer for whole of Belgium ---
    water_gdf = gpd.read_file(osm_file)

    # --- 4. Reproject if necessary ---
    if water_gdf.crs != tile_gdf.crs:
        water_gdf = water_gdf.to_crs(gr.crs)
        
    # --- Quick bbox filter (fast)
    water_bbox = water_gdf.cx[xmin:xmax, ymin:ymax]

    # --- 5. Clip this bbox (which is the same as the grid exent).
    clipped_gdf = gpd.clip(water_bbox, tile_gdf)

    # --- 6. Save if requested ---
    if output_file is not None:
        clipped_gdf.to_file(output_file)

    print(f"Tile bounds: X={gr.x[0]}-{gr.x[-1]}, Y={gr.y[-1]}-{gr.y[0]}")
    print(f"Original features: {len(water_gdf)}, Clipped features: {len(clipped_gdf)}")

    return clipped_gdf

def line_length_per_gr_cell(gr, clipped_gdf):
    """
    Compute total length of poly_lines per grid cell.
    
    Parameters
    ----------
    gr: Grid object
        grid object holding the grid coordinates
    clipped_gdf : GeoDataFrame
        Clipped waterlines (LineString or MultiLineString), same CRS as clipped_gdf.
    
    Returns
    -------
    grid : GeoDataFrame
        Grid cells with total water length per cell (column 'water_length_m').
        Each record is a grid cell with its coordinates and the surface water length in it.
    """    

    # --- 1. Grid coordinates are in the grid object ---
    x = gr.x
    y = gr.y
    
    # --- 1. Cell boxes as a comprehension
    # --- Generate a list of shapely.geometry.boxes, around each model cell
    boxes = [box(xL, yB, xR, yT)                    
                    for yB, yT in zip(y[1:], y[:-1])
                        for xL, xR in zip(x[:-1], x[1:])]
    
    # --- Turn this into a geoDataFrame.
    grid_cells_gdf = gpd.GeoDataFrame(geometry=boxes, crs=clipped_gdf.crs)
    
    # --- Add a column with the global cell index in the grid    
    grid_cells_gdf['I'] = np.asarray(gr.NOD[0].flatten(), dtype=int)

    # --- 2. Overlay waterlines with grid cells ---
    intersections = gpd.overlay(clipped_gdf, grid_cells_gdf, how="intersection")

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
        # --- Convert point tuple to GeoDataFrame
        pt_geom = Point((gr.x.mean(), gr.y.mean()))  # point = (x, y) in EPSG:31370
        pt_gdf = gpd.GeoDataFrame(geometry=[pt_geom], crs=gr.crs)

        # --- Plot
        tile_gdf = get_tile_gdf(gr)
        
        # --- Plot tile boundary and return the axes.
        ax = tile_gdf.boundary.plot(color="red", linewidth=2, figsize=(8,8))
        
        # --- Plot the surface waters clipped to the tile
        clipped_gdf.plot(ax=ax, color="blue", linewidth=1)
        
        # --- Plot points
        pt_gdf.plot(ax=ax, color="black", markersize=50)

        ax.set_title("15x15 km modelgebied met oppervlaktewater")
        
        ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'tile_plot.png'))

        # --- New plot. Plot a grid for orientation and return the axes.
        ax = grid_gdf.plot(column="water_length_m", cmap="Blues", legend=True, figsize=(8,8))
        
        # --- Plot the clipped surface waters
        clipped_gdf.plot(ax=ax, color="black", linewidth=0.5)
        
        # --- Add the points
        pt_gdf.plot(ax=ax, color='red', markersize=50)
        
        ax.set_title("grid tile with density of water features per cell")

        ax.figure.savefig(os.path.join(os.getcwd(), 'images', 'surf_wat_in_cells.png'))
        
    return grid_gdf # Just use column ["water_length"]

def select_15x15km_from_surfwater_belgium(xy=(194000, 195000)):    
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

def climb_to_ridge(x:float, y:float, dist: Raster=None)->np.ndarray:
    """Climb from current pixel to ridge.
    
    Parameters
    ----------
    x, y: starting point indices
    dist: raster
        raster with distance to surface water
    """
    ix, iy = ~dist.transform(x, y)
    path = [(ix, iy)]

    # --- First step direction unknown.
    ndy_prev, ndx_prev = None, None

    ny, nx = dist.array.shape

    while True:
        """ --- Step until the ridge of the landscape is reached."""
        current = dist.raster[iy, ix]

        # --- 3×3 window
        iy0, iy1 = max(iy-1, 0), min(iy+2, ny)
        ix0, ix1 = max(ix-1, 0), min(ix+2, nx)
        window = dist[iy0:iy1, ix0:ix1]

        ndy, ndx = np.unravel_index(np.argmax(window), window.shape)
        iy_new = iy0 + ndy
        ix_new = ix0 + ndx

        # -- new step direction
        ndy_step = iy_new - iy
        ndx_step = ix_new - ix

        if ndy_prev is not None:
            # --- look forward in the same direction
            iy_fwd = iy + ndy_prev
            ix_fwd = ix + ndx_prev

            if (0 <= iy_fwd < ny and 0 <= ix_fwd < nx and
                dist[iy_fwd, ix_fwd] <= current):
                    # --- cross maximum reached (so we're on the ridge)
                    break

        ndy_prev, ndx_prev = ndy_step, ndx_step
        iy, ix = iy_new, ix_new
        path.append((ix, iy))
        
    path = np.array(path, dtype=int)
    
    return dist.transform * (path.T[0], path.T[1])

def get_distance_raster(water_gdf, pixelsize=10):
    """Return raster with distance to surface water."""
    
    # --- get bounding box coordinates for water_gdf
    xmin, ymin, xmax, ymax = water_gdf.total_bounds
    
    # --- Compute the shape of the raster
    n_width = int((xmax - xmin) / pixelsize)
    n_height = int((ymax - ymin) / pixelsize)
    
    # --- Generate an affine coordinate transform from N, W and pixelsize
    transform = from_origin(xmin, ymax, pixelsize, pixelsize)
    
    # --- Generate the raster
    #     The burn-in value is used for points of the geom shapes. The rest are fill (background)
    burn_in_value = 1
    fill = 0
    raster = features.rasterize(
        [(geom, burn_in_value) for geom in water_gdf.geometry],  # your geometries with a burn value
        out_shape=(n_height, n_width),               # shape of the raster
        transform=transform,                         # affine transform for your raster
        fill=fill,                                      # value for background
        dtype = 'uint8'        
    )
    
    # --- Generate raster with shortest distance to the burn in value.
    #     That is the shortest distance from the raster point to points with value 1.
    dist_to_water = distance_transform_edt(burn_in_value - raster) * pixelsize
    
    return Raster(dist_to_water, transform=transform, crs=None, nodata=fill)
    
   
def get_ridge_mask(dist_to_water: Raster, slope_thresh=0.5)->Raster:
    """Return a Raster mask object marking the grates
    
    Parameters
    ----------
    dist_to_water: Raster class object
        Raster holding distance to water and other metadata.
    """
    ny, nx = dist_to_water.shape
    
    ridgemask = dist_to_water.copy()
    ridgemask.array = np.zeros_like(dist_to_water.array, dtype=bool)
    
    pixelsize = dist_to_water.pixelsize
    
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):

            template = dist_to_water.array[iy-1:iy+2, ix-1:ix+2]
            NW, N, NE, W, C, E, SW, S, SE = template.ravel()

            is_ridge = False

            # --- horizontale graat
            if (W < C) and (E < C):
                S_grate = abs(S - N) / (2 * pixelsize)
                is_ridge = True

            # --- verticale graat
            elif (N < C) and (S < C):                
                S_grate = abs(E - W) / (2 * pixelsize)
                is_ridge = True

            # --- diagonaal ↘
            elif (NW < C) and (SE < C):                
                S_grate = abs(NE - SW) / (2 * np.sqrt(2) * pixelsize)
                is_ridge = True

            # --- diagonaal ↗
            elif (NE < C) and (SW < C):                
                S_grate = abs(NW - SE) / (2 * np.sqrt(2) * pixelsize)
                is_ridge = True

            if is_ridge:
                ridgemask.array[iy, ix] = (S_grate < slope_thresh)
                   
    return ridgemask

def cleanup_simplify(xy=None, L=15000):
    """Return a gdf with cleaned and simplified Belgian surface water.
    
    Parameters
    ----------
    xy: tuple of floats.
        Coordinaters within the Belgian coordinate system: "EPSG:31370".
        Centre of the area to clip.
    L: float
    Size of the area to clip.
    """
    xm, ym = xy
    x, y, z = [xm - L/2, xm + L/2], [ym - L/2, ym + L/2], [0, -1]
    
    # --- Generate a simple Modflow grid consisting of a single cell.
    gr = Grid(x, y, z)
    gr.crs = "EPSG:31370"
    
    # --- Clip of the surface water to this cell.
    clip_gdf = clip_water_to_gr(gr)
    
    # --- Generate of column with the lengt of each surface water line.
    # clip_gdf = clip_gdf.explode(ignore_index=True)
    clip_gdf["length_m"] = clip_gdf.length
    
    # --- Throw out the records of lines smaller than min_length
    min_length = 250 # m
    water_main = clip_gdf[clip_gdf["length_m"] >= min_length].copy()
    
    # --- Reindex
    water_main = clip_gdf.explode(index_parts=False).reset_index(drop=True)

    # --- Plot the remaining surface water lines.
    ax = water_main.plot(color="lightblue", linewidth=1)
    
    ax.set_title("Water_main")

    # --- Report the number of line records
    print("Voor:", len(clip_gdf))
    print("Na  :", len(water_main))
    
    # --- Plot the remaining lines on new figure    
    ax = clip_gdf.plot()
    ax.set_title("Original, uncleaned")
    
    # --- Plot the cleaned surface waters on new figure
    ax = water_main.plot()
    ax.set_title("without small loose pieces")

    # --- Simplify the surface water given a tolerance.
    # --- And plot on new figure
    water_simpl = water_main.copy()
    water_simpl["geometry"] = water_simpl.geometry.simplify(
            tolerance=100, preserve_topology=True
    )
    ax = water_simpl.plot()
    ax.set_title("Simplified surface water")
    
    # --- Generate a raster (pixelsize) with the distance to surface water
    #     Using the simplified surface water gdf.
    dist_to_water = distance_raster(water_main.geometry, pixelsize=10)
    
    pixelsize = dist_to_water.pixelsize
    transform = dist_to_water.transform
    
    nrows, ncols = dist_to_water.shape
    
    left = transform.c
    top =  transform.f
    right = left + ncols * transform.a
    bottom= top  + nrows * transform.e
    
    x = left + np.arange(ncols) * pixelsize
    y = top  - np.arange(nrows) * pixelsize
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Distance to Surface Water')

    # --- Coordinaten van waarnemingsputten
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

    # --- Contour the distance to water
    contours = ax.contour(x, y, dist_to_water, levels=20, colors='black', linewidths=0.5, origin='upper')
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    
    # --- Plot the surface water on top
    water_simpl.plot(ax=ax)
    ax.set_title("Voorbeeld opp.water met afstandskaart en denkbeeldige waarnemingsputten")
    # ax.plot(*obs_coordinates.T, 'ro', label="observation points")
    
    # --- Plot the observation points on top
    # --- Transform the obs coordinates to the grid coordinages (indices)
    xi, yi = ~transform * (obs_coordinates[:, 0], obs_coordinates[:, 1])
    xi, yi = np.asarray(xi, dtype=int), np.asarray(yi, dtype=int)
    
    # --- Compute the nearest ridgepoint for each observation point
    obs_ridge = []
    bs = []
    for x, y in obs_coordinates:
        ix, iy = ~transform * (x, y)
        
        # --- Climb to the ridge
        ix_, iy_ = climb_to_ridge(ix, iy, dist_to_water)
        x_, y_ = transform * (float(ix_), float(iy_))
        
        bs.append(dist_to_water[iy_, ix_])
        obs_ridge.append((x_, y_))
        
    obs_ridge = np.array(obs_ridge, dtype=float)    
    xx = np.column_stack([obs_coordinates[:, 0], obs_ridge[:, 0]]).T
    yy = np.column_stack([obs_coordinates[:, 1], obs_ridge[:, 1]]).T

    # --- Plot the ridge points
    ax.plot(xx, yy, '.-', label='climb')
    print("xDupuit = ", np.round(np.sqrt(((obs_coordinates - obs_ridge) ** 2).sum(axis=1))))
    print("bDupuit = ", np.round(np.array(bs)))
    
def distance_grid_with_grates(xy=None, L=15000, dxy=50):
    
    xm, ym = xy()
    xmin, ymin, xmax, ymax = xm - L/2, ym - L/2, xm + L/2, ym + L/2
    
    gr = Grid(x=np.linspace(xmin, xmax, nx+1), y=np.linspace(ymin, ymax, ny+1), z=[0, -1])
    gr.crs="EPSG:31370"
    water_gdf = clip_water_to_gr(gr)
    dist_to_water = get_distance_raster(water_gdf, pixelsize=dxy)
    ridge_mask = get_ridge_mask(dist_to_water)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(title="Dist to water", xlabel='x[m]', ylabel='y [m]')
    
    max_dist = np.ceil(dist_to_water.array.max() * 100) / 100
    C = ax.contour(gr.xm, gr.ym, dist_to_water.array, linewidths=0.5, colors='k', levels=np.arange(0, max_dist, 100))
    ax.clabel(C, levels=C.levels)
            
    ax.imshow(ridge_mask.array, origin='upper', alpha=0.2, extent=gr.extent)
    
    water_gdf.plot(ax=ax, color='blue')

    ax.set_title('Afstand tot oppervlaktewater en kamverloop (geel)') 
    
    fig.savefig(os.path.join(images, "afstand_tot_oppwater.png"))
    
    dist_to_water.ridge_mask = ridge_mask.array
    return dist_to_water 
        
        
# --- Coordinaten van waarnemingsputten
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

# %%
if __name__ == '__main__':
    xy = (194000, 195000)
    L, dxy = 15000, 50
    (xm, ym), nx, ny = xy, int(L/dxy), int(L/dxy)
    xmin, ymin, xmax, ymax = xm - L/2, ym - L/2, xm + L/2, ym + L/2
    
    if False:
        select_15x15km_from_surfwater_belgium(xy=xy)
    if False:
        cleanup_simplify(xy=xy, L=L)
    if True:
        dist_to_water = distance_grid_with_grates(xy=xy, L=L, dxy=dxy)
        ax = plt.gca()
        for obs in obs_coordinates:
            path = climb_to_ridge(obs)[[0, -1]]
            ax.plot(path.T, 'r-')
            ax.plot(*path[0], 'o', ms=10, mec='r', mfc='none')
            ax.plot(*path[1], 'o', ms=10, mec='b', mfc='none')
 
    plt.show()
    
# %%
