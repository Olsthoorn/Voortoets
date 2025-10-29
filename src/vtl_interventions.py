# %%
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import shape
from shapely.geometry import shape as shapely_shape
from glob import glob
from pathlib import Path
from zipfile import ZipFile
import etc
from flopy.mf6.utils import MfGrdFile
from fdm.src.mfgrid import Grid
from src.vtl_layering import get_layering

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


def userinput_fr_json(zip_folder):
    """Return GeoDataFrame with userinfo of all cases in project_foler."""
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
    

def intervention_cases_fr_json(project_folder):
    """Load all groundwater intervention cases into a dict keyed by simulation_name."""
    
    # --- userinput.json files
    jsons = [f + '/input/userinput.json'
             for f in [folder for folder in
                        glob(project_folder + '/Rap*')
                        if not folder.endswith('.zip')]
             ]
    cases = {}
    for id, json_file in enumerate(jsons):
        with open(json_file, "r") as f:
            data = json.load(f)

        sim_name = data.get("simulation_name")
        if not sim_name:
            raise ValueError(f"{json_file} has no 'simulation_name' field.")

        # Parse all 'shape' JSON strings into shapely geometries
        for section in ("interventions", "receptors"):
            if section not in data:
                continue
            for subtype, items in data[section].items():
                for item in items:
                    if "shape" in item:
                        try:
                            geojson_obj = json.loads(item["shape"])
                            item["geometry"] = shapely_shape(geojson_obj)
                            item.pop("shape")
                        except Exception as e:
                            print(f"Warning: could not parse shape in {json_file}: {e}")
        
        cases[id] = data

    return cases

        

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
    for s in interventions_gdf['simulation_name']:
        pName_sType = s_to_pName_sType(s=s)
        pname_simtype.append(pName_sType)
    # --- Return without first flattening
    # return list(set(pname_simtype))
    return pname_simtype


def grid_from_interventions(interventions, L=15000., tsmult=1.25, show=False):
    """Return grid generated one cases interventions.
    
    Parameters
    ----------
    interventions: cases[id]['interventions
        each case as a list of interventions, each of which as list of
        intervention objects that each have a geometry.
        All the geometries are combined to generate a model grid.
        The layering of the center of the grid is then acquired and
        simply added as an attribute to the grid for layer use.
    L: float
        size in both directions of the model to be generated
    tsmult: float
        factor by which dx and dy increases beyond grid central part
        which is defined by the boundary around the geometries
    """
    def floor(x): return np.floor(x / 10) * 10
    def ceil(x):  return np.ceil( x / 10) * 10
    
    def r_series(N=100, L=15000, tsmult=1.1):
        """Return series with dr increasing by  factor tsmult"""
        dr = np.ones(N) * 10.
        for i in range(1, len(dr)):
            dr[i] = dr[i-1] * tsmult
        r = np.cumsum(dr)[np.cumsum(dr) < L / 2]
        return r

    r = r_series(100, L=L, tsmult=tsmult)

    intervs = []
    for key in interventions:
        intervs += interventions[key]

    # --- get linestrings coordinates
    coords = []
    for interv in intervs:
        if isinstance(interv['geometry'], shapely.Point):
            coords.append(np.array([[interv['geometry'].x, interv['geometry'].y]]))
        else:
            coords.append(np.array(interv['geometry'].coords.xy).T)
    coords = np.vstack(coords)
    
    # --- get  x and y coordinates
    x, y = coords.T
    
    # --- bounding box adjusted to nicer national grid values
    xmin, ymin, xmax, ymax = floor(x.min()), floor(y.min()), ceil(x.max()), ceil(y.max())

    # --- central part of grid with constant dx and dy
    x = np.linspace(xmin, xmax, int((xmax - xmin) / 10.) + 1)
    y = np.linspace(ymin, ymax, int((ymax - ymin) / 10.) + 1)
    xc, yc = x.mean(), y.mean()
    
    # --- rel coords
    xr, yr = x - xc, y - yc

    # --- extend rel coordinates with r series on both sides
    x = np.unique(np.hstack((xr[0] - r[::-1], xr, xr[-1] + r, -L/2, L/2)))
    y = np.unique(np.hstack((yr[0] - r[::-1], yr, yr[-1] + r, -L/2, L/2)))
    
    # --- cut off to size of local model and convert to national coordinates
    xGr = x[np.logical_and(x >= -L/2, x <= L/2)] + xc
    yGr = (y[np.logical_and(y >= -L/2, y <= L/2)] + yc)[::-1]
    
    layering = get_layering(shapely.Point(xc, yc))
    zGr = layering['z']
    
    gr = Grid(xGr, yGr, zGr, axial=False)
    gr.layering = layering
    
    # --- show the grid
    if show:
        ax = etc.newfig("Grid from lines", "x", "y")
        ax.vlines(xGr, yGr.min(), yGr.max())
        ax.hlines(yGr, xGr.min(), xGr.max())
    
    return gr # With layering
    
    

        

    
    
    # --- generate a grid GeoDataFrame from the cells

    # --- Intersect the lines with the grid

    # --- Get the grid id s and length within the cells
    
    # --- Compute the conductances

    # --- Complete the riv or separage GHB
    

# %%
try:
    zipfolder = os.path.join(os.getcwd(), '../data', '6194_GWS_testen')
    assert os.path.isdir(zipfolder), f"Path not found <{zipfolder}>"
except Exception as e:
    zipfolder = os.path.join(os.getcwd(), 'data', '6194_GWS_testen')
    assert os.path.isdir(zipfolder), f"Path not found <{zipfolder}>"
    
# %%
# --- Example usage:
interventions_gdf = userinput_fr_json(zipfolder)
print(interventions_gdf.head())
print(f"Loaded {len(interventions_gdf)} features from {interventions_gdf['simulation_name'].nunique()} projects")

# %%
# intervections_dict = get_interventions_dict(interventions_gdf)

cases = intervention_cases_fr_json(project_folder=zipfolder)

# %%
intervs = []
for id in range(len(cases)):
    intervs += cases[id]['interventions'].keys()
intervs = [str(s) for s in np.unique(intervs)]
intervs

def get_interventions(case):
    """Return the intervections for case with given caseId"""

    interventions = case['interventions']
    for key in interventions:
        if key == 'dewatering_line': # lowering, LINESTRING
            dewat_lines = interventions[key]
            dwl_dict = {}
            for dwl in dewat_lines:
                for lowering in dwl.lowering:
                    t, s = lowering
                    
                    if not t in dwl_dict:
                        dwl_dict[t] = {}
                        dwl_dict[t]['geometry'] = []
                        dwl_dict[t]['s'] = []
                        
                    dwl_dict[t]['coords'].append(dwl['geometry'])
                    well_dict[t]['s'].append(s)
            return dwl_dict                    

        elif key == 'dewatering_polygon': # lowering, POLYGON
            pgons = interventions[key]
            pgons_dict = {}
            for p in pgons:
                for lowering in p['lowerings']:
                    t, s = lowering
                    
                    if not t in pgons_dict:
                        pgons_dict[t] = {}
                        pgons_dict[t]['coords'] = p['geometry']
                        pgons_dict[t]['s'] = s
                        
                    dwl_dict[t]['coords'].append(w['geometry'].x, w['geometry'].y)
                    well_dict[t]['Q'].append(Q)
                    well_dict[t]['depth_aquifer'] = w['depth_aquifer']

                        
            return pgons_dict
        elif key == 'extraction_general_point': # flow_rates, depth_aquifer, POINT
            wells = interventions[key]  
            well_dict = {}
            for w in wells:
                for flow_rate in w['flow_rates']:
                    t, Q = flow_rate
                    
                    if not t in well_dict:
                        well_dict[t] = {}
                        well_dict[t]['coords'] = []
                        well_dict[t]['Q'] = []
                        
                    well_dict[t]['coords'].append(w['geometry'].x, w['geometry'].y)
                    well_dict[t]['Q'].append(Q)
                    well_dict[t]['depth_aquifer'] = w['depth_aquifer']
            return(well_dict)          
        elif key == 'extraction_irrigation_point': # flow_rates, depth_filter, POINT
            wells = interventions[key]
            well_dict = {}
            for w in wells:
                for flow_rate in w['flow_rates']:
                    t, Q = flow_rate
                    
                    if not t in well_dict:
                        well_dict[t] = {}
                        well_dict[t]['coords'] = []
                        well_dict[t]['Q'] = []
                        
                    well_dict[t]['coords'].append(w['geometry'].x, w['geometry'].y)
                    well_dict[t]['Q'].append(Q)
                    well_dict[t]['depth_aquifer'] = w['depth_aquifer']
            return(well_dict)
        elif key == 'hardening_polygon': # percentage, POLYGON
            pgons = interventions[key]
            pgons_dict = {}
            for p in pgons:
                for tp in pgons['percentage']:
                    t, perc = tp

                    if not t in pgons_dict:
                        pgons_dict[t] = {}
                        pgons_dict[t]['percentage'] = []
                        
                    pgons_dict[t]['geometry'] = p['geometry']
                    pgons_dict[t]['per'].append(perc)
                    pgons_dict[t]['name'] = p['name']
            return pgons_dict
        elif key == 'recharge_point': # flow_rates, depth_aquifer, POINT
            rech_wells = interventions[key]
            well_dict = {}
            for w in rech_wells:
                for flow_rate in w.flow_rates:
                    t, Q = flow_rate

                    if not t in well_dict:
                        well_dict[t] = {}
                        well_dict[t]['coords'] = []
                        well_dict[t]['Q'] = []

                    well_dict[t]['coords'].append(w['geometry'].x, w['geometry'].y)
                    well_dict[t]['Q'].append(Q)
            return(well_dict)
        else:
            raise ValueError(f"Can't get here, key ={key}")
                        
intervs = []
for id in range(len(cases)):
    interv += list(cases[id]['interventions'].keys())
intervs
# %% 
        
# %%
# --- add columns project name and the simulation type to the gdf
pname_simtype = get_simulation_types(interventions_gdf)
interventions_gdf['p_name'] = [p[0] for p in pname_simtype]
interventions_gdf['s_type'] = [p[1] for p in pname_simtype]
interventions_gdf = interventions_gdf.drop(['id', 'simulation_name'], axis=1)
print("Added columns 'p_name' and 's_type'")
print("Done")

# --- show which projects or layer_types have flow rates defined
mask = interventions_gdf["flow_rates"].isna()
np.unique(interventions_gdf.loc[~mask, ['layer_type']])

# --- show which projects or layer_types have lowering rates defined
mask = interventions_gdf["lowering"].isna()
np.unique(interventions_gdf.loc[~mask, ['layer_type']])

# --- show which s_types there are
s_types = np.unique(interventions_gdf["s_type"])


# --- show which projects or layer_types have neither "flow_rates" nor "lowering" defined
mask = np.logical_and(interventions_gdf["flow_rates"].isna(), interventions_gdf["lowering"].isna())
np.unique(interventions_gdf.loc[~mask, ['layer_type']])

# --- show which projects or layer_types have neither "flow_rates" nor "lowering" defined
mask = interventions_gdf["s_type"] == "filterbemaling"
fbm = interventions_gdf.loc[~mask, ['p_name', 's_type', 'layer_type']]
print(fbm)


# %%
gdfs = {}
s_types = np.unique(interventions_gdf["s_type"])
for s_type in s_types:
    mask = interventions_gdf['s_type'] == s_type
    gdfs[s_type] = interventions_gdf.loc[mask].dropna(axis=1, how='all')
    
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
