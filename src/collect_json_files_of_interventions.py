# %%
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from pathlib import Path
from zipfile import ZipFile
import etc
from flopy.mf6.utils import MfGrdFile

# %%
def project_to_gdf(project_json):
    """Convert one project JSON dict to a GeoDataFrame."""
    def to_records(features, layer_type):
        recs = []
        for f in features:
            geom = shape(json.loads(f["shape"]))
            rec = {k: v for k, v in f.items() if k != "shape"}
            rec.update({
                "layer_type": layer_type,
                "geometry": geom,
            })
            recs.append(rec)
        return recs

    sim_name = project_json.get("simulation_name", "unknown_project")

    records = []
    for group in ["interventions", "receptors"]:
        if group in project_json:
            for layer_type, features in project_json[group].items():
                records.extend(to_records(features, layer_type))

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:31370")
    gdf["simulation_name"] = sim_name
    return gdf


def load_projects_from_zipfolder(zip_folder):
    """Read all userinput.json files from ZIPs in folder into one GeoDataFrame."""
    zip_folder = Path(zip_folder)
    gdfs = []

    for zip_path in zip_folder.glob("*.zip"):
        with ZipFile(zip_path) as zf:
            # find the correct file (case-insensitive just in case)
            json_candidates = [n for n in zf.namelist() if n.lower().endswith("input/userinput.json")]
            if not json_candidates:
                print(f"⚠️  No userinput.json found in {zip_path.name}")
                continue
            json_path = json_candidates[0]
            with zf.open(json_path) as f:
                project_data = json.load(f)
            gdf = project_to_gdf(project_data)
            gdf["source_zip"] = zip_path.name
            gdfs.append(gdf)

    if not gdfs:
        print("No valid projects found.")
        return None

    return gpd.pd.concat(gdfs, ignore_index=True)
    # return gdfs


def s_to_pName_sType(s="projectNm-id-simType"):
    """Return project name and simulation type s.
    
    Parameters
    ----------
    s: str with form 'pName-nr' + '-' + simType 
        string to be split
    """
    
    match = re.match(r'^((?:[^-]*-){2}?)(.*)$', s)

    if match:
        pName = match.group(1).rstrip('-')
        sType = match.group(2)
    # print("Before second hyphen:", pName)
    # print("After second hyphen:", sType)
    return (pName, sType)

def get_simulation_types(projects_gdf):
    """Return simulation types.
    
    Parameters
    ----------
    projects.gdf: geopandas DataFrame
        gdf extracted from userinput.json of all projects simulated by AGT
    
    Returns
    -------
    Unique names of simulation types obtained from
    gdf['simulation_name']
    """
    pname_simtype = []
    for s in all_projects_gdf['simulation_name']:
        pName_sType = s_to_pName_sType(s=s)
        pname_simtype.append(pName_sType)
    # --- Return without first flattening
    # return list(set(pname_simtype))
    return pname_simtype


# %%
try:
    zipfolder = os.path.join(os.getcwd(), '../data', '6194_GWS_testen')
    assert os.path.isdir(zipfolder), f"Path not found <{zipfolder}>"
except Exception as e:
    zipfolder = os.path.join(os.getcwd(), 'data', '6194_GWS_testen')
    assert os.path.isdir(zipfolder), f"Path not found <{zipfolder}>"
    

# --- Example usage:
all_projects_gdf = load_projects_from_zipfolder(zipfolder)
print(all_projects_gdf.head())
print(f"Loaded {len(all_projects_gdf)} features from {all_projects_gdf['simulation_name'].nunique()} projects")

# --- add columns project name and the simulation type to the gdf
pname_simtype = get_simulation_types(all_projects_gdf)
all_projects_gdf['p_name'] = [p[0] for p in pname_simtype]
all_projects_gdf['s_type'] = [p[1] for p in pname_simtype]
all_projects_gdf = all_projects_gdf.drop(['id', 'simulation_name'], axis=1)
print("Added columns 'p_name' and 's_type'")
print("Done")

# --- show which projects or layer_types have flow rates defined
mask = all_projects_gdf["flow_rates"].isna()
np.unique(all_projects_gdf.loc[~mask, ['layer_type']])

# --- show which projects or layer_types have lowering rates defined
mask = all_projects_gdf["lowering"].isna()
np.unique(all_projects_gdf.loc[~mask, ['layer_type']])

# --- show which s_types there are
s_types = np.unique(all_projects_gdf["s_type"])


# --- show which projects or layer_types have neither "flow_rates" nor "lowering" defined
mask = np.logical_and(all_projects_gdf["flow_rates"].isna(), all_projects_gdf["lowering"].isna())
np.unique(all_projects_gdf.loc[~mask, ['layer_type']])

# --- show which projects or layer_types have neither "flow_rates" nor "lowering" defined
mask = all_projects_gdf["s_type"] == "filterbemaling"
fbm = all_projects_gdf.loc[~mask, ['p_name', 's_type', 'layer_type']]
print(fbm)



# %%
gdfs = {}
s_types = np.unique(all_projects_gdf["s_type"])
for s_type in s_types:
    mask = all_projects_gdf['s_type'] == s_type
    gdfs[s_type] = all_projects_gdf.loc[mask].dropna(axis=1, how='all')
    
for s_type in s_types:
    print(s_type)
    print(gdfs[s_type][['p_name', 'name']])
    print()

# %%
"""
Hieronder staan de simulation_types met de kenmerkende eigenschappen die voor de modellering van belang zijn.

s_types:
--------
0. bronbemaling                dewatering_polygon="POLYGON (...)", lowering=[[0, 3.95]]
1. filterbemaling              dewatering_polygon="POLYGON (...)", lowering=[[0, 3.95], [183, 0], [365, 0]]
2. lijnbemaling-filters        dewatering_line=LINESTRING (...), lowering=[[0, 1.37], [14,0], [84, 0]]
    heeft meerdere dewatering_lines met eigen "naam" zoals "moot1", "moot2"
3. open-bemaling               layer_type="dewatering_polygon", lowering=[[0, 1], [183, 0]]
4. permanente-bemaling         layer_type="dewatering_polygon", lowering=[[0, 1]]
5. permanente-winning          layer_type="extraction_general_point" flow_rates=[[0, 500000]], putten hebben een aparte "name" zoasl "put1", "put2", ...
6. retourbemaling-bronnen (bronbemaling + retourputten) dewatering_polygon with lowering maar ook recharge_points with flow_rates en met elke put een "name" zoals "IP1", "IP2", ... Is maar een project: lc 216
7. seizoenale-winning          layer_type="extraction_irrigation_point", flow_rates=[[0, 20000]], depth_filter{'top': 7, 'bottom': 17} of depth_aquifer"A0800"
8. seizoenale-winning-20k      layer_type="extraction_irrigation_point", flow_rates=[[0, 20000]], depth_aquifer="A0170"
9. verharding                  layer_type="hardening_polygon",  percentage=[[0, 100]], p_name="bm4"
"""

# %%

s_type = s_types[9]

gdf = gdfs[s_type]

l_types = gdf['layer_type']


p_names = np.unique(gdf["p_name"])
for p_name in p_names:
    ax = etc.newfig(f"{p_name} {s_type}", f"x [m], {gdf.crs}", f"y [m] {gdf.crs}")
    mask = gdf["p_name"] == p_name
    p_gdf = gdf.loc[mask]
    clrs = etc.color_cycler()
    for l_type in l_types:
        clr = next(clrs)
        mask = p_gdf['layer_type'] == l_type
        p_gdf.loc[mask].plot(ax=ax, ec=clr, fc='none', linewidth=1)    
    ax.set_aspect(1.0)


# %% --- get grid info from the grb file

# --- the csv file of the vertices was extracted from the .disv file on the same directory
grb_file = os.path.join(zipfolder, 'Rapport LC_212_filterbemaling', 'model', 'no_permits.disv.grb')
assert os.path.isfile(grb_file), f"Can't open file <{grb_file}>"

# --- read the grid
grb = MfGrdFile(grb_file)

# --- Get some basic info
print(grb.grid_type)        # should say 'DISV'
print(grb.ncpl)             # number of cells
print(grb.nlay)             # number of layers

# --- Get cell coordinates and vertices
xy = grb.cellcenters

verts  = grb.verts           # list of (x, y) vertex coordinates
iverts = grb.iverts          # cell-to-vertex connectivity

# --- top of the model
grb.top

# --- elevation of layer bottoms
botm = grb.bot.reshape(grb.nlay, grb.ncpl)

# --- get the center of the model (near the point of the project)
dxy = [[x, y]] - grb.verts
R = np.sqrt(dxy.T[0] + dxy.T[1])
i = np.argmin(R)
z = np.hstack((grb.top[i], botm[:, i]))

# %%
# Get the conductivities
# Get the storage coefficients
# Get the stress periods
