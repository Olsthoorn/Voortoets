# %% [markdown]
# # Get the layerering and layer properties of an arbitrary location in Flanders
#
# The grid and detail information of the three regional models of the Flamish
# "grondwatersimulator" are in the coding/data/regional_grids folder.
# Next to the three grids (as .grb files), the directory also contains the
# info about the properties of these models in a geopackage "regiona_models.gpkg"
# together, these files can provide the layering and propertes at any point
# within Flanders.
#
# TO 2025-10-27
#

# %%

import os
import numpy as np
import geopandas as gpd
from shapely import Point
from flopy.mf6.utils import MfGrdFile
from glob import glob

# %% --- regional model grids and folder

# --- regional model grids
grb_files = {'N': 'region_north.disv.grb',
             'W': 'region_west.disv.grb',
             'E': 'region_east.disv.grb'
            }

crs = "EPSG:31370"

# --- regional model grids folder
try:
    reg_folder = os.path.join(os.getcwd(), '../data', 'regional_grids') 
    assert os.path.isdir(reg_folder), f"No folder <{reg_folder}>"
except Exception:
    reg_folder = os.path.join(os.getcwd(), 'data', 'regional_grids')
    assert os.path.isdir(reg_folder), f"No folder <{reg_folder}>"

try:
    prj_folder = os.path.join(os.getcwd(), '../data', '6194_GWS_testen') 
    assert os.path.isdir(prj_folder), f"No folder <{prj_folder}>"
except Exception:
    prj_folder = os.path.join(os.getcwd(), 'data', '6194_GWS_testen')
    assert os.path.isdir(prj_folder), f"No folder <{prj_folder}>"



# %%

def select_reg_model(point, center=True):
    """Return regional model grid based on given point"""
    
    # --- has crs 'EPSG:31370'
    gdf2 = gpd.read_file(os.path.join(reg_folder, 'regional_models.gpkg'))
    
    # --- point in convec hull of model grids
    inside_mask = gdf2.contains(point)
    inside = gdf2[inside_mask]

    if inside.empty:
        print("Point is not inside any polygon.")
    else:
        if len(inside) == 1:
            # Only one polygon contains the point
            selected = inside.iloc[0]
        else:
            # Multiple polygons contain the point
            
            inside = inside.copy()
            # --- Option A: closest centroid ---
            if center:                
                inside["centroid_dist"] = inside.centroid.distance(point)
                selected = inside.loc[inside["centroid_dist"].idxmin()]
            else:
                # --- Option B: farthest from polygon boundary ---
                # Compute distance to the polygon's exterior boundary
                inside["boundary_dist"] = inside.boundary.distance(point)
                selected = inside.loc[inside["boundary_dist"].idxmax()]

        # --- Return selected pd.core.series.Series
        selected.crs = str(gdf2.crs)
        return selected

        print("Selected record:")
        print(selected)


def get_layering(pnt, center=True):
    """Return layering of subsurface at Point ctr.
    
    Parameters
    ----------
    pnt: shapely.Point
        Point at which the layering is desired (center of fur)
    center: bool to choose regional model of more than one contains pnt
        True:  pnt is closest to centroid of regional model
        False: pnt is farthest from convec hull of regional model
    """
    
    def s2arr(s):
        """Convert string to float array."""
        return np.fromstring(s.strip('[]'), sep=' ')
    
    # --- all geological layers ---
    geo_layers =np.array([
                'A0100', 'A0210', 'A0220', 'A0230', 'A0240', 'A0250',
                'A0300', 'A0400', 'A0500', 'A0600', 'A0700', 'A0800',
                'A0900', 'A1000', 'A1100'])
    
    # --- select regional model containing pnt
    reg_mdl = select_reg_model(pnt, center=center)
    
    # --- get its Modflow disv grid
    grb = MfGrdFile(os.path.join(reg_folder, grb_files[reg_mdl['index']]))
    
    # --- cell centers of grid
    xcenters, ycenters = grb.cellcenters.T
    xP, yP = pnt.x, pnt.y
    
    # --- get cel with center closest to pnt
    dist2 = (xcenters - xP) ** 2 + (ycenters - yP) ** 2
    idx_flat = np.argmin(dist2)

    # --- reshape btm to [nlay, ncpl]    
    botm = grb.bot.reshape(grb.nlay, grb.ncpl)
    
    # --- all layers
    z = np.vstack((grb.top, botm))[:, idx_flat]
    
    # --- active layers at this point
    mask = z[1:] < z[:-1]
    
    # --- layering for active layers only
    layering = {
        'model': reg_mdl['model'],               
        'x': xP,
        'y': yP,
        'z': np.hstack((z[0], z[1:][mask])),    # layer elev at pnt
        'layers': geo_layers[:len(mask)][mask],
        'k'   : s2arr(reg_mdl['k'])[mask],
        'k33' : s2arr(reg_mdl['k33'])[mask],
        'ss'  : s2arr(reg_mdl['ss'])[mask],
        'sy'  : s2arr(reg_mdl['sy'])[mask],
        'crs' : reg_mdl.crs,
        'nlay_orig': len(botm),
    }
    return layering

# %%
if __name__ == '__main__':
    gdf2 = gpd.read_file(os.path.join(reg_folder, 'regional_models.gpkg'))

    pnt = Point(150000, 200000)

    selected = select_reg_model(pnt, center=True)
    selected = select_reg_model(pnt, center=False)

    layering = get_layering(pnt)

# %%

shps = []
for folder in glob(prj_folder + '/*'):
    shps = glob(folder + '/sourcedata/*.shp')
    layers = np.unique([os.path.basename(f)[:5] for f in shps])
    print(os.path.basename(folder) + '[' + ', '.join(layers) + ']')


# %%
