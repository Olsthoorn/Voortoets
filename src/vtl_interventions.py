# %%
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.getcwd()).parent))

import json
import re
import numpy as np
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
from src import vtl_regional as reg

# %%
try:
    prj_folder = os.path.join(os.getcwd(), '../data', '6194_GWS_testen')
    assert os.path.isdir(prj_folder), f"Path not found <{prj_folder}>"
except Exception as e:
    prj_folder = os.path.join(os.getcwd(), 'data', '6194_GWS_testen')
    assert os.path.isdir(prj_folder), f"Path not found <{prj_folder}>"

# %%
def project_to_gdf(project_json):
    """Convert one project JSON dict to a GeoDataFrame.
    
    Is now obsolete. Use userinput_fr_json instead to get all in dict.
    
    """
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
    

def cases_fr_json(prj_folder):
    """Load all groundwater intervention cases into a dict keyed by simulation_name."""
    
    # --- userinput.json files
    jsons = [f + '/input/userinput.json'
             for f in [folder for folder in
                        glob(prj_folder + '/Rap*')
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


def sim_name_to_pName_sim_type(s="projectNm-id-simType"):
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
        elif isinstance(interv['geometry'], shapely.Polygon): 
            x, y = interv['geometry'].exterior.coords.xy
            coords.append(np.vstack((x, y)).T)
        elif isinstance(interv['geometry'],shapely.LineString):
            x, y = np.array(interv['geometry'].coords.xy)
            coords.append(np.vstack((x, y)).T)
        else:
            raise ValueError("Can't get here!")
        
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
    
    layering = reg.get_layering(shapely.Point(xc, yc))
    zGr = layering['z']
    
    gr = Grid(xGr, yGr, zGr, axial=False)
    gr.layering = layering
    gr.crs = layering['crs']
    
    # --- show the grid
    if show:
        ax = etc.newfig("Grid from lines", "x", "y")
        ax.vlines(xGr, yGr.min(), yGr.max())
        ax.hlines(yGr, xGr.min(), xGr.max())
    
    return gr # With layering


def chd_fr_dewatering_polygons(gr, pgons, t_end=365):
    """Convert dewatering polygon into a CHD object."""
    dtypeCHD = np.dtype([('t', float), ('I', int), ('h', float)])
    
    CHD_list = []
    for pgon in pgons:
        mask = gr.inpoly(np.array(pgon['geometry'].exterior.coords.xy).T)
        Id = gr.NOD[0][mask]
        
        lowerings = pgon['lowering']
        if lowerings[-1][0] < t_end / 2:
            lowerings.append([np.ceil(t_end / 2), 0])
        if lowerings[-1][0] < t_end:
            lowerings.append([t_end, 0])
        
        for lowering in lowerings:
            t, s = lowering
            chd = np.zeros(len(Id), dtype=dtypeCHD)
            chd['t'] = t
            chd['I'] = Id            
            chd['h'] = -s
            CHD_list.append(chd)
    CHD_list = np.vstack(CHD_list)
    
    CHD = {}
    for t in np.unique(CHD_list['t']):
        mask = CHD_list['t'] == t
        CHD[t] = CHD_list[mask]
    return CHD


def wel_fr_hardening_polygons(gr, pgons, recharge=0.001, t_end=365):
    """Convert hardening polygons into a WEL object to be handled by FDM.
    
    Each hardening_polygons is interpreted as extraction of a percentation 
    the net recharge over that polygon."""
        
    dtypeWEL = np.dtype([('t', float), ('I', int), ('Q', float)])
    
    HPC_list = []
    for pgon in pgons:
        pg = np.array(pgon['geometry'].exterior.coords.xy).T
        mask = gr.inpoly(pg)
        Id = gr.NOD[0][mask]
        
        percentages = pgon['percentage']
        if percentages[-1][0] < t_end:
            percentages.append([t_end, 0])
        
        for percentage in percentages:
            t, perc = percentage
            hpc = np.zeros(len(Id), dtype=dtypeWEL)
            hpc['t'] = t
            hpc['I'] = Id            
            hpc['Q'] = -gr.Area.ravel()[Id] * recharge * perc
            HPC_list.append(hpc)
    HPC_list = np.vstack(HPC_list)
    
    WEL = {} # Hardening is converted to WEL in cells
    for t in np.unique(HPC_list['t']):
        mask = HPC_list['t'] == t
        WEL[t] = HPC_list[mask]
    return WEL


def wel_fr_recharge_points(gr, rch_points, t_end=365):
    WEL = wel_fr_well_points(gr, rch_points, t_end=t_end)
    # --- flow rates are injection not extraction
    for k in WEL:        
        WEL[k]['Q'] = -WEL[k]['Q']
    return WEL


def wel_fr_extraction_general_points(gr, well_points, t_end=365):
    return wel_fr_well_points(gr, well_points, t_end=t_end)


def wel_fr_extraction_irrigation_points(gr, well_points, t_end=365):
    return wel_fr_well_points(gr, well_points, t_end=t_end)

    
def wel_fr_well_points(gr, well_points, t_end=365):
    """Convert well_points into a WEL object to be handled by FDM."""
    dtypeWEL = np.dtype([('t', float), ('I', int), ('Q', float)])
    
    WEL_list = []
    for well in well_points:
        xw, yw = well['geometry'].x, well['geometry'].y
        
        # --- get zw (generally it's the first aquifer center)
        z = gr.layering['z']
        zm = 0.5 * (z[:-1] + z[1:])        
        aquif = [k for k in well if 'aquifer' in k]
        depth = [k for k in well if 'depth'   in k]     
        if aquif and aquif[0] in well:
            laynm = well[aquif[0]]
            if laynm.startswith('A'):
                laynm = laynm[:-1] + '0' # corrects error in the user input
                try:
                    ilay = np.where(laynm == gr.layering['layers'])[0][0]
                except Exception:
                    ilay = np.argmax(-np.diff(z))
                zw = zm[ilay]
        elif depth and depth[0] in well:
            top = z[0] -well[depth[0]]['top']
            bot = z[0] -well[depth[0]]['bottom']
            zw = 0.5 * (top + bot)
        else:
            zw =zm[0]
            
        Id = gr.Iglob_from_xyz(np.array([[xw, yw, zw]]))
        
        flow_rates = well['flow_rates']
        if flow_rates[-1][0] < t_end / 2:
            flow_rates.append([np.ceil(t_end / 2), 0])
        if flow_rates[-1][0] < t_end:
            flow_rates.append([t_end, 0])
    
        for flow_rate in flow_rates:
            t, Q = flow_rate
            wel = np.zeros(1, dtype=dtypeWEL)
            wel['t'] = t
            wel['I'] = Id
            wel['Q'] = -Q / 365. # m3/a to m3/d
            WEL_list.append(wel)
    
    WEL_list = np.vstack(WEL_list)
        
    WEL = {}
    for t in np.unique(WEL_list['t']):
        mask = WEL_list['t'] == t
        WEL[t] = WEL_list[mask]
    return WEL


def chd_fr_dwatering_lines(gr, dewatering_lines, t_end=365):
    """Convert dewatering lines into a CHD object to be handled by FDM."""
    dtypeCHD = np.dtype([('t', float), ('I', int), ('h', float)])
    ds = 1.0
    
    CHD_list = []
    for dewatering_line in dewatering_lines:
        line = dewatering_line['geometry']
        
        # --- generate intermediate points at mutual distance of 1 m
        # --- s along linestring
        points = [line.interpolate(d) for d in np.arange(0, line.length, ds)]
        xyz = np.array([(p.x, p.y, gr.zm[0]) for p in points])

        # --- cell Id's of all intermediate points
        Id = np.unique(gr.Iglob_from_xyz(xyz))
        
        # --- determine number of subpoints in each cell (rather total length in each cell)
        # L = []
        # for id in Id:
        #    --- total length in cell id
        #    # L.append(ds * np.sum(Iall == id))
        # L = np.array(L)
        
        # --- add the lowerings
        lowerings = dewatering_line['lowering']
        if lowerings[-1][0] < t_end / 2:
            lowerings.append([np.ceil(t_end / 2), 0])
        if lowerings[-1][0] < t_end:
            lowerings.append([t_end, 0])
             
        for lowering in lowerings:
            t, s = lowering
            chd = np.zeros(len(Id), dtype=dtypeCHD)
            chd['t'] = t
            chd['I'] = Id
            chd['h'] = -s
            CHD_list.append(chd)
            
        CHD_list = np.vstack(CHD_list)
        CHD = {}
        for t in np.unique(CHD_list['t']):
            mask = CHD_list['t'] == t
            CHD[t] = CHD_list[mask]
        return CHD


def get_geological_layers_per_model():
    """Study which geological layers are in each project's model"""
    # reg_folder = os.path.join(prj_folder, '../regional_grids')
    # gdf2 = gpd.read_file(os.path.join(reg_folder, 'regional_models.gpkg'))
    
    cases = cases_fr_json(prj_folder=prj_folder)
    for id, case in cases.items():
        rcep = case['receptors']['receptor_point'][0]['geometry']
        layering = reg.get_layering(rcep, center=True)
        sim_name = case['simulation_name']
        case_dir = ('Rapport ' + sim_name[:2].upper() + sim_name[2:]).replace('-','_')
        case_dir = os.path.join(prj_folder, case_dir, 'sourcedata')
        # print(os.path.isdir(case_dir), case_dir)
        laynms = np.unique([os.path.basename(f)[:5] for f in glob(case_dir + '/*.shp')])
        laynms = [str(n) for n in laynms if n.startswith('A')]
        print(f"{layering['model']} original {layering['nlay_orig']} layers: [{', '.join(layering['layers'])}]")
        
        

# %%
if __name__ == '__main__':
    pass
        
    # %%
    # --- Example usage:

    # --- is now obsolete
    interventions_gdf = userinput_fr_json(prj_folder)
    print(interventions_gdf.head())

    print(f"Loaded {len(interventions_gdf)} features from {interventions_gdf['simulation_name'].nunique()} projects")

    # %%
    # %% [markdown]
    # # Overview of all cases with their interventions
    # All cases are read in from their userinput.json files (<case folder>/input/usrinput.json)
    # They are listed below together with their intervention types and number of interventions objects
    # like dewatering_polygons and wells (recharge_points, extraction_irrigation_points etc.)

    # %%
    # --- get user input for all cases from the project folder. The userinput is a json file
    # --- which determines case and its interventions.

    cases = cases_fr_json(prj_folder=prj_folder)

    # --- Show for each case which interventions is has:
    print('\n# --- interventions ---\n')

    for id in range(len(cases)):
        case = cases[id]
        
        s1 = f"case {id} {case['simulation_name']:35}: "

        interventions = case['interventions']
        s2 = []
        for k in interventions.keys():
            s2.append(f"{k}: N={len(interventions[k])}")
        print(s1 + ', '.join(s2))


    print("\n# --- Reception points ---\n")

    proj_coords = []

    for id in range(len(cases)):
        case = cases[id]
        
        s1 = f"case {id} {case['simulation_name']:35}: "
        
        rps = case['receptors']['receptor_point']
        s2 = []
        for rp in rps:
            x, y =  [rp['geometry'].x, rp['geometry'].y]
            proj_coords.append([x, y])        
            s2.append(f"{rp['name']:20}, x={x:6.0f}, y={y:6.0f}")

        print(s1 + ', '.join(s2))
        
    proj_coords = np.array(proj_coords)
    ax = etc.newfig("Project_locations", "xB", "yB")

    for id in range(len(proj_coords)):
        angle = id * 3.
        ax.plot(*proj_coords[id], 'ro')
        x, y = proj_coords[id]
        dx, dy = np.cos(5000), np.sin(5000)
        ax.text(x + dx, y + dy, cases[id]['simulation_name'], rotation=angle, ha='left', va='bottom')

    # There are in fact only 5 really different case locations

    ax.plot(*proj_coords.T, 'or', label='project_locations')
    ax.legend()


    # %% [markdown]
    # # intervention object data
    # Each case has interventions defined by first the type of intervention objects, which
    # is followed by a list of actual intervention objects with individual properties.
    # As such, a building pit has "dewatering_polygon" as the type of its intervention objects.
    # (there is no case having more than one "dewatering_objects", but new case might have them).
    # Cases with lijnbemaling have "dewatering_line" as their intervention object type, which is
    # then followed by as list of actual "dewatering_line" objects. Each such object may have
    # its own lowering schedule in time and lowewring defined by its parameter "lowering" which is
    # a list of [time, lowering] pairs.
    # Likewise for wells (permanente winning, irrigatie, retourbemaling)
    # Cases specifying retourbemaling have two intervention types:
    #    1) dewatering_polygon (followed by a list of (1) dewatering polyogons)
    #    2) recharge_point (followed by a list of n recharge points) with flow_rates.
    # Projects defining hardening of a surface have "hardening_polygon" as their object type
    # followed by a list of (1) such objects, each of which has its geometry and its percentage of hardening over time (list of [time, perc] pairs).

            
    # %% --- get grid info from the grb file

    # --- the csv file of the vertices was extracted from the .disv file on the same directory
    grb_file = os.path.join(prj_folder, 'Rapport LC_212_filterbemaling', 'model', 'no_permits.disv.grb')
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
