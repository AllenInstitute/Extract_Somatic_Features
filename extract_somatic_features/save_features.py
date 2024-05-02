from extract_somatic_features.feature_collection import get_feat_dict
import gcsfs
from taskqueue import queueable
import json
import io
import os
import h5py
from cloudfiles import CloudFiles
from caveclient import CAVEclient



@queueable
def write_feat_json(cell_info,
                  nuc_seg_source,
                  gcloud_filepath,
                  voxel_resolution,
                  fixmesh_input,
                  dataset_name = '',
                  mat_version = None,
                  update = False,
                  token_file=None):
    """for each cell will collection the soma and nucleus reature dictionary and save it as a 
    json to the specified google bucket folder
    """
    
    nuc_id = cell_info[0]
    soma_id = cell_info[1]
    
    print(f'Starting Cell {soma_id}')
    matversion = mat_version
    caveclient = CAVEclient(dataset_name)
    mesh_dict, soma_mesh = get_feat_dict(cell_info,
                  nuc_seg_source = nuc_seg_source,
                  voxel_resolution= voxel_resolution,
                  fixmesh_input = fixmesh_input,
                  caveclient = caveclient,
                  mat_version = matversion,
                  get_nucleus = True,
                  return_mesh = True)
    
    fs = gcsfs.GCSFileSystem(project = 'em-270621', token = os.environ.get('GCSFS_TOKEN', token_file))
    google_bucket = 'gs://allen-minnie-phase3'
    soma_filepath = os.path.join(google_bucket, 'minniephase3-somas-v661/')
    soma_filename = '%d.h5'%(soma_id)

    cf = CloudFiles(soma_filepath)

    bio = io.BytesIO()

    with h5py.File(bio, "w") as f:
        f.create_dataset("vertices", data=soma_mesh.vertices, compression="gzip")
        f.create_dataset("faces", data=soma_mesh.faces, compression="gzip")

    cf.put(
        soma_filename,
        content=bio.getvalue(),
        content_type="application/x-hdf5",
        compress=None,
        cache_control=None,
    )
            
    cell_json = json.dumps(mesh_dict, indent = 4) 
    json_filepath = os.path.join(gcloud_filepath, 'minniephase3-somafeatures-v661/')
    jsonfilename = json_filepath + '%d_%d.json'%(nuc_id, soma_id)
    #if updating, ensure the file already exists
    exists = fs.exists(jsonfilename)
    if update==True:
        assert(exists == True)
        with fs.open(jsonfilename, 'r') as f:
            current_json = f.read(f)
        current_json.update(cell_json)
        with fs.open(jsonfilename, 'w') as f:
            f.write(current_json)
    
    else:
        assert(exists == False)
        with fs.open(jsonfilename, 'w') as f:
            f.write(cell_json)


    print('%d features updated'%(nuc_id))