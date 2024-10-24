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
def write_feat_json(cell_info,nuc_seg_source,gcloud_filepath,voxel_resolution,fixmesh_input,dataset_name = '',mat_version = None,update = False,token_file=None):
    """
    Writes the features of a cell to a JSON file.
    Args:
        cell_info (tuple): A tuple containing the nucleus ID and soma ID of the cell.
        nuc_seg_source (str): The source of the nucleus segmentation.
        gcloud_filepath (str): The path to the Google Cloud Storage directory where the JSON file will be saved.
        voxel_resolution (float): The voxel resolution of the cell.
        fixmesh_input (str): The input for fixing the mesh.
        dataset_name (str, optional): The name of the dataset. Defaults to an empty string.
        mat_version (str, optional): The version of the mat file. Defaults to None.
        update (bool, optional): Whether to update an existing JSON file. Defaults to False.
        token_file (str, optional): The path to the token file. Defaults to None.
    Raises:
        AssertionError: If update is True and the JSON file does not exist.
    Returns:
        None
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


@queueable
def write_local_json(cell_info, nuc_seg_source, out_path, voxel_resolution, fixmesh_input, dataset_name='', mat_version=None):
    """
    Write the features of a cell as a JSON file.
    Parameters:
    - cell_info (tuple): A tuple containing the nucleus ID and soma ID of the cell.
    - nuc_seg_source (str): The source of the nucleus segmentation.
    - out_path (str): The path to save the JSON file.
    - voxel_resolution (float): The voxel resolution.
    - fixmesh_input (str): The input for fixing the mesh.
    - dataset_name (str, optional): The name of the dataset. Defaults to an empty string.
    - mat_version (str, optional): The version of the mat file. Defaults to None.
    Returns:
    - None
    Raises:
    - None
    """
    
    
    nuc_id = cell_info[0]
    soma_id = cell_info[1]
    
    print(f'Starting Cell {soma_id}')
    mat_version = mat_version
    client = CAVEclient(dataset_name)
    mesh_dict, soma_mesh = get_feat_dict(cell_info,
                nuc_seg_source = nuc_seg_source,
                voxel_resolution= voxel_resolution,
                fixmesh_input = fixmesh_input,
                caveclient = client,
                mat_version = mat_version,
                get_nucleus = True,
                return_mesh = True)

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

    jsonfilename = '%d_%d.json'%(nuc_id, soma_id)
    json_filepath = os.path.join(out_path, jsonfilename)

    with open(json_filepath, 'w') as f:
        json.dump(cell_json, f, ensure_ascii=False, indent=4)


    print('%d features updated'%(nuc_id))