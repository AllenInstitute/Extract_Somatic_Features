from extract_somatic_features.utils import *
import os
import gcsfs
from extract_somatic_features.Fix_mesh import FixMesh

def get_current_file_ids(filetype, filepath, gcsfs_project):
    """
    Retrieves the file IDs for the files of a specific type in a given filepath.

    Parameters:
    - filetype (str): The type of files to search for.
    - filepath (str): The path to the directory containing the files.
    - gcsfs_project (str): The project name for the GCSFileSystem.

    Returns:
    - file_ids (list): A list of integers representing the file IDs.
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
    """
    Retrieves the file IDs for a given file type in a specified filepath.

    Parameters:
    - filetype (str): The file type to search for.
    - filepath (str): The path to the directory containing the files.

    Returns:
    - file_ids (list): A list of integers representing the file IDs.

    Example:
    >>> get_local_file_ids('.txt', '/path/to/files')
    [1, 2, 3]
    """

    filenames = os.listdir(filepath)

    file_ids = []
    for f in filenames:
        if filetype in f:
            cellid = f.split(filetype)[0]
            cellid = cellid.split('_')[-1]
            file_ids.append(int(cellid))
    return file_ids

def load_mesh(segid, cv_path, remove_duplicates=False):
    """
    Load a mesh from a given segid using the specified cv_path.

    Args:
        segid (int): The ID of the segment to load.
        cv_path (str): The path to the cv file.
        remove_duplicates (bool, optional): Whether to remove duplicate vertices. Defaults to False.

    Returns:
        trimesh.base.Trimesh: The loaded mesh.

    """

    meshmeta = trimesh_io.MeshMeta(cv_path=cv_path)
    mesh = meshmeta.mesh(seg_id = segid,
                         remove_duplicate_vertices=remove_duplicates)
    print("loaded %d"%(segid))

    if mesh.is_watertight==False:
        mesh.fix_mesh()

    return mesh

def load_fixed_mesh(segid, ctr_pt_nm, fix_input_json):
    """
    Load a fixed mesh using the given parameters.
    Args:
        segid (str): The segment ID.
        ctr_pt_nm (str): The control point name.
        fix_input_json (dict): The input JSON for fixing the mesh.
    Returns:
        tuple: A tuple containing the fixed mesh and the fraction of zero values.
    """


    mod = FixMesh(**fix_input_json)
    fixed_mesh, frac_zero = mod.fix(segid, ctr_pt_nm=ctr_pt_nm)
    
    return fixed_mesh, frac_zero


def get_json_gcsfs(cellid, filename, project_name):
    """
    Retrieves a JSON file from Google Cloud Storage using GCSFileSystem.

    Args:
        cellid (tuple): A tuple containing the nucid and somaid of the cell.
        filename (str): The path to the JSON file in Google Cloud Storage.
        project_name (str): The name of the Google Cloud project.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """

    nucid = cellid[0]
    somaid = cellid[1]
    fs = gcsfs.GCSFileSystem(project = project_name)
    with fs.open(filename, 'rb') as file:
        cell_json = json.load(file)
    return cell_json

