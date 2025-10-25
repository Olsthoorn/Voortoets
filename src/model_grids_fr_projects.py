
# %% [markdown]
# # Find out which regional models the Flamish "Grondwatersimulator" heeft.
#
# The .grb files are used to get the cell vertices and around
# each model a convex hull is generated.
# All hulls go into a geodataframe with is then plotted.
#
# The results show that there are only three regional models.
#
# TO 2025-10-24


# %% 
import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from flopy.mf6.utils import MfGrdFile
from glob import glob
import etc

# %%

def alpha_shape(points, alpha):
    """Compute a concave hull (alpha shape) of a set of points."""
    if len(points) < 4:
        raise ValueError("To few points.")

    coords = np.array(points)
    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    def add_edge(i, j):
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = coords[ia], coords[ib], coords[ic]
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = (a + b + c) / 2.0
        area = max(s * (s - a) * (s - b) * (s - c), 0.0)
        circum_r = a * b * c / (4.0 * np.sqrt(area)) if area > 0 else np.inf
        if circum_r < 1.0 / alpha:
            add_edge(ia, ib)
            add_edge(ib, ic)
            add_edge(ic, ia)

    m = MultiPoint(coords)
    m = unary_union([Polygon(e) for e in edge_points])
    polygons = list(polygonize(m))
    return unary_union(polygons)


def contour_from_points(points, res=100):
    x, y = np.array(points).T
    xi = np.linspace(x.min(), x.max(), res)
    yi = np.linspace(y.min(), y.max(), res)
    grid = np.zeros((res, res))
    ix = np.searchsorted(xi, x)
    iy = np.searchsorted(yi, y)
    grid[iy, ix] = 1
    cs = plt.contour(xi, yi, grid, levels=[0.5])
    polys = [Polygon(seg) for seg in cs.allsegs[0]]
    return unary_union(polys)

def load_model_grids_from_zipfolder(projects_folder):
    """Read all userinput.json files from ZIPs in folder into one GeoDataFrame."""
    
    pr_names = glob(projects_folder + '/*.zip')
    pr_names = [p.replace('.zip', '') for p in pr_names]
    
    grds = []

    for pr_name in pr_names:
        grb_file = os.path.join(pr_name, 'regional', 'no_permits.disv.grb')
        assert os.path.isfile(grb_file), f"No file <{grb_file}>"
            
        print(f"Processing <{os.path.basename(os.path.basename(pr_name))}>")
    
        grd = MfGrdFile(grb_file)
        points = MultiPoint(grd.verts)
        
        # Compute convex hull
        hull = MultiPoint(points).convex_hull
        
        # alpha = 100.        
        # hull = alpha_shape(grd.verts, alpha)
        
        #hull = contour_from_points(grd.verts, res=500)


        reg_grd = gpd.GeoDataFrame(geometry = [hull])
        reg_grd['project'] = os.path.basename(pr_name)
        reg_grd['model_type'] = 'regional'        
        reg_grd.crs="EPSG:31370"
        
        grds.append(reg_grd)

    if not grds:
        print("No valid projects found.")
        return None

    grd_gpd = gpd.pd.concat(grds, ignore_index=True)
    # --- reorder columns
    
    grd_gpd['xC'] = np.round(grd_gpd.geometry.centroid.x)
    grd_gpd['yC'] = np.round(grd_gpd.geometry.centroid.y)

    # Centroids models W, N and E approximately
    mdl_centroids = {
         'W': {'x':  91500., 'y': 183800.},
         'N': {'x': 181200., 'y': 206100.},
         'E': {'x': 193300., 'y': 168800.}
        }

    mask_W = np.isclose(grd_gpd['xC'], mdl_centroids['W']['x'], atol=300.)
    mask_N = np.isclose(grd_gpd['yC'], mdl_centroids['N']['y'], atol=300.)
    mask_E = np.isclose(grd_gpd['xC'], mdl_centroids['E']['x'], atol=300.)
    
    # --- which regional model (W, E or N) we have
    grd_gpd['model'] = 'any'
    grd_gpd.loc[mask_N, 'model'] = 'N'
    grd_gpd.loc[mask_E, 'model'] = 'E'
    grd_gpd.loc[mask_W, 'model'] = 'W'
    
    cols = [c for c in grd_gpd.columns if c != 'geometry'] + ['geometry']
    
    return grd_gpd[cols] 


# %%
try:
    projects_folder = os.path.join(os.getcwd(), '../data', '6194_GWS_testen')
    assert os.path.isdir(projects_folder), f"Path not found <{projects_folder}>"
except Exception as e:
    projects_folder = os.path.join(os.getcwd(), 'data', '6194_GWS_testen')
    assert os.path.isdir(projects_folder), f"Path not found <{projects_folder}>"

gdf_regional_model_hull = load_model_grids_from_zipfolder(projects_folder)

reg_models_center = gdf_regional_model_hull.geometry

ax = etc.newfig("Convec hull of all regional models", "xB", "yB")
gdf_regional_model_hull.plot(ax=ax, fc='none', ec='b', linewidth=0.25)

gdf_regional_model_centroid = gdf_regional_model_hull.copy()
gdf_regional_model_centroid.geometry = gdf_regional_model_hull.geometry.centroid

gdf_regional_model_centroid.plot(ax=ax, color='r')

print('Done')
    
# %% [markdown]
# # Conclusion
#
# There are three regional models as proven by the convec hulls.
# We may call them 'W', 'E' and 'NE'
# The centers of these models can be given from the picture
# "W"  (90000, 190000)
# "E"  (200000, 160000)
# "NE" (180000, 220000)
# Which model to use depending on project location?
# 1. Inside the hull
# 2. If inside more than one hull, the one whose center is closest to the project location.

# reg_grds = load_model_grids_from_zipfolder(zip_folder)

# %%
