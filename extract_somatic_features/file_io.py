from extract_somatic_features.utils import *
import os
import gcsfs
from extract_somatic_features.Fix_mesh import FixMesh

def get_current_file_ids(filetype, filepath, gcsfs_project):
    """returns a list of all the cell IDs from existing files 
    in the given folder. Expects the directory to be on google storage.
    """
    fs = gcsfs.GCSFileSystem(project = gcsfs_project)
    filenames = fs.ls(filepath)

    file_ids = []
    for f in filenames:
        if filetype in f:
            cellid = f.split(filetype)[0]
            cellid = cellid.split('_')[-1]
            file_ids.append(int(cellid))
    return file_ids

def get_local_file_ids(filetype, filepath):
    """returns a list of all the cell IDs from existing files 
    in the given folder. Expects a local directory.
    """
    filenames = os.listdir(filepath)

    file_ids = []
    for f in filenames:
        if filetype in f:
            cellid = f.split(filetype)[0]
            cellid = cellid.split('_')[-1]
            file_ids.append(int(cellid))
    return file_ids

def load_mesh(segid, 
              cv_path,
              remove_duplicates=False):
    """loads and returns a mesh downloaded directy from the given cloudvolume segmentation source. 
    If the mesh is not watertight will attempt to make it watertight.
    """
    meshmeta = trimesh_io.MeshMeta(cv_path=cv_path)
    mesh = meshmeta.mesh(seg_id = segid,
                         remove_duplicate_vertices=remove_duplicates)
    print("loaded %d"%(segid))

    if mesh.is_watertight==False:
        mesh.fix_mesh()

    return mesh

def load_fixed_mesh(segid,
                    ctr_pt_nm,
                    fix_input_json):
    """fixes the mesh of the input segid according to the parameters in the fix_input_json
    and returns the mixed mesh
    """

    mod = FixMesh(**fix_input_json)
    fixed_mesh, frac_zero = mod.fix(segid, ctr_pt_nm=ctr_pt_nm)
    
    return fixed_mesh, frac_zero

def get_json_gcsfs(cellid,filename,project_name):
    """returns the feature json for the given cell
    """
    nucid = cellid[0]
    somaid = cellid[1]
    fs = gcsfs.GCSFileSystem(project = project_name)
    with fs.open(filename, 'rb') as file:
        cell_json = json.load(file)
    return cell_json

