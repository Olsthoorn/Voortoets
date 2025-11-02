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
from pprint import pprint

# %% --- regional model grids and folder

# --- regional model grids
grb_files = {'N': 'region_north.disv.grb',
             'W': 'region_west.disv.grb',
             'E': 'region_east.disv.grb'
            }

# --- all geological layers found in the 3 Flamish regional models---
all_geo_layers =np.array([
            'A0100', 'A0210', 'A0220', 'A0230', 'A0240', 'A0250',
            'A0300', 'A0400', 'A0500', 'A0600', 'A0700', 'A0800',
            'A0900', 'A1000', 'A1100'])

# --- crs of Belgium, for all coordinates
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

def s2arr(s):
    """Convert string like '[12. 23. ...]' to float array."""
    return np.fromstring(s.strip('[]'), sep=' ')

    
def get_reg_models_as_dict():
    """Return thei info of the three Flamish regional models as a dict."""
    
    # --- read regional models gneal data stored as geopackage in to geopandas.GeoDataFrame
    reg_gdf = gpd.read_file(os.path.join(reg_folder, 'regional_models.gpkg'))
    
    # --- Cleanup and convert to dictionary
    reg_gdf.index = reg_gdf['index']
    reg_dict = reg_gdf.drop(columns=['index']).T.to_dict()
    
    # --- Convert the layer properties from strings to float arrays
    for mdl_id in reg_dict: # mdl_id: ['W', 'N', 'E']
        for k in ['k', 'k33', 'ss', 'sy']:
            reg_dict[mdl_id][k] = s2arr(reg_dict[mdl_id][k])
            
        # --- Add names of geological layeres"
        reg_dict[mdl_id]['nlay'] = len(reg_dict[mdl_id]['k'])
        reg_dict[mdl_id]['geo_layers'] = all_geo_layers[:reg_dict[mdl_id]['nlay']]
    
    return reg_dict



def select_reg_model(point, center=True):
    """Return regional model grid based on given point"""
    
    # --- has crs 'EPSG:31370'
    reg_gdf = gpd.read_file(os.path.join(reg_folder, 'regional_models.gpkg'))
    
    # --- point in convec hull of model grids
    inside_mask = reg_gdf.contains(point)
    inside = reg_gdf[inside_mask]

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
        selected.crs = str(reg_gdf.crs)
        return selected


def get_layering(point, center=True):
    """Return layering of subsurface at Point ctr.
    
    Parameters
    ----------
    point: shapely.Point
        Point at which the layering is desired (center of fur)
    center: bool to choose regional model of more than one contains point
        True:  point is closest to centroid of regional model
        False: point is farthest from convec hull of regional model
    """
            
    # --- select regional model containing point
    reg_mdl = select_reg_model(point, center=center)
    
    # --- get its Modflow disv grid
    grb = MfGrdFile(os.path.join(reg_folder, grb_files[reg_mdl['index']]))
    
    # --- cell centers of grid
    xcenters, ycenters = grb.cellcenters.T
    xP, yP = point.x, point.y
    
    # --- get cel with center closest to point
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
        'z': np.hstack((z[0], z[1:][mask])),    # layer elev at point
        'layers': all_geo_layers[:len(mask)][mask],
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
    pass

    # %% [markdown]
    # # Regional models info
    # The "grondwatersimulator" of Flanders has 3 regioinal models. The info about these models
    # such as their grid, geological layers and layer properties with centroid and convex hull
    # has been gathered from the regional models used by the intervention cases obtained
    # in the context of the Voortoets project. This info was then save to a geopackage file
    # which can be read in by geopandas, yielding a geopandas.GeoDataFrame.
    # In a second step, this geodataframe is converted to a dictionary for the
    # info about the three regional models, using 'W', 'N' and 'E' as their indentifying key.
    # Finally the geological names of the layers of these models are added to the dictionary.

    # %%
    reg_gdf  = gpd.read_file(os.path.join(reg_folder, 'regional_models.gpkg'))
    reg_gdf
    
    # %% [markdown]
    # ### The regional model dictionary:
    reg_dict = get_reg_models_as_dict()
    
    for id in reg_dict:
        print(f"REGIONAL MODEL with id = {id}")
        print("------------------------------")
        
        pprint(reg_dict[id])
        print()

    # %% [markdown]
    # # Retrieving the layering an properties at an arbitrary location
    # It's now easy to get the regional model and the layering and layer properties
    # at an arbitrary location.
    # lThe ayering at the location is also obtained from the info of the regional model
    # selected for that location. The number of layers and the layer names are a subset of
    # those of the entire regional model, because not all layers are present everywhere.
    # The regional model works such that it has all geologcal layers everywhere, however
    # if, in reality, a layer is not locally present, its layer thickness is zero.
    # This allows selecting the layer elevations, sequence and names that are relevant
    # for the particular location. 

    # --- arbitrary location
    point = Point(150000, 200000)

    # --- select the regional model for this location (use center=True or False)
    # --- if center is True the regional model is chosen for which the distance to its convex hull is maximum, else the regional model is chosen for which the distance to its centroid is minimum.
    selected = select_reg_model(point, center=True)
    selected = select_reg_model(point, center=False)

    # --- The layering and layer properties at an arbitrary location can be obtained like so
    # --- Notice that within get_layering select_regional model is selected first
    layering = get_layering(point)
    layering

    # %% [mardown]
    # # Geological layers used within each project/case
    # Each project/case uses one of the three flamish regional models. The project's
    # sourcedata folder contains the shapefiles that were used. The first 5 characters
    # of each shapefile name shows with geological layers were used in that project.
    # Not all projects using the same regonal model have the same number of layers,
    # probably because locally not all layers are present.
    # By going through all projects and for each project get the names of the shapefiles
    # in the sourcedata folder and only use the first 5 characters of these filenames,
    # and sorting them reveals all geological layers that were actually used in each project.
    # The series of names are consistent, between projects, so that the longest list
    # in any project using one of the regional models reveails the total list of
    # all geological layers in the regional model. In fact the list turns out
    # to be consistent even between the regional models. Hence the longest list
    # in any project is the list of all geologcal layers, if the length of
    # this list is equal to the length of the number of layers in any regional model.
    # This turns out to be the case, the largest number of layers in any of the three
    # regional models equals 15.

    all_layers = set()
    for folder in glob(prj_folder + '/*'):
        shps = glob(folder + '/sourcedata/*.shp')
        layers = np.unique(
            [os.path.basename(f)[:5] for f in shps if os.path.basename(f).startswith('A')]
        )
        all_layers = all_layers.union([str(L) for L in layers])
        
        print(f"{os.path.basename(folder)[8:].replace('.zip',""):30} [{' '.join(layers)}]")
        
    print("All layers:\n----------")
    print(all_layers)

# %%
