import sys
sys.path.append("../")
from extract_somatic_features.feature_collection import get_feat_dict
import json
# import pandas as pd
# from multiprocessing import Pool
# from functools import partial
from caveclient import CAVEclient
from meshparty import trimesh_io
import json
import numpy as np
import os
#import h5py
#from cloudfiles import CloudFiles

#Input file with all the dataset specific information
inputfile = '../data/test_input.json'
f = open(inputfile)
args = json.load(f)
chunksize = args['chunk']
nucleus_table = args['nucleus_table'] #tablename with nucleus pts for all cells 
dataset = args['dataset']
mat_version = args['materialization_version']
out_path = args['output_filepath']

#Setting up client to access public dataset
client = CAVEclient(dataset)
cv_path = client.info.segmentation_source()

#Parameters to clean and remesh the soma cutout
fixmesh_input = {
    'image_source':client.info.image_source(),
    "cv_path" : cv_path,
    'disk_cache_path': args['disk_cache_path'],
    'mip_level':args['mip_level'],
    'cutout_radius':args['cutout_radius']
    }


#example cell from public dataset version 661
sample_nuc_id = 154146
#Looking up the pt position and soma segmentation id from the nucleus table
nuc_lookup = client.materialize.query_view(nucleus_table,
                                            materialization_version=mat_version)

df = nuc_lookup.query('id == @sample_nuc_id')
sample_soma_id = df.pt_root_id.values[0]
cell_info = (sample_nuc_id, sample_soma_id, df.pt_position_lookup.values[0])


print(f'Starting Cell with Nucleus ID {sample_nuc_id}')
print(cell_info)
#Extracting nucleus and soma features
mesh_dict, soma_mesh = get_feat_dict(cell_info,
                nuc_seg_source = args['nuc_seg_source'],
                voxel_resolution= args['voxel_resolution'],
                fixmesh_input = fixmesh_input,
                caveclient = client,
                mat_version = mat_version,
                get_nucleus = True,
                return_mesh = True)

#Saving the fixed somatic mesh as an H5 file
soma_filepath = os.path.join(out_path, 'meshes/')
soma_filename = 'sample_%d.h5'%(sample_soma_id)
soma_filepath = os.path.join(soma_filepath, soma_filename)
# Using cloudfiles to save the mesh components


trimesh_io.write_mesh_h5(soma_filename, soma_mesh.vertices,soma_mesh.faces)

#Ensure proper encoding
class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

# Saving the features as a json       
cell_json = json.dumps(mesh_dict, cls=json_serialize)

jsonfilename = '%d_%d.json'%(sample_nuc_id, sample_soma_id)
json_filepath = os.path.join(out_path, jsonfilename)

with open(json_filepath, 'w') as f:
    json.dump(cell_json, f, ensure_ascii=False, indent=4)
    