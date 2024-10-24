import numpy as np
from extract_somatic_features.utils import *
from extract_somatic_features.file_io import *


def get_mesh_features(mesh_id, mesh, 
                      caveclient = None,
                      mat_version = None,
                      ctr_pt_nm = [], 
                      voxel_resolution = [4,4,40],
                      soma=False):
    """
    Returns a dictionary with geometric mesh features. If the mesh is a soma, 
    Parameters:
    - mesh_id (int): The ID of the mesh.
    - mesh (Mesh): The geometric mesh object.
    - caveclient (optional): The caveclient object used for querying synapse data. Default is None.
    - mat_version (optional): The materialization version for querying synapse data. Default is None.
    - ctr_pt_nm (list): The coordinates of the center point in nanometers. Default is an empty list.
    - voxel_resolution (list): The voxel resolution in nanometers. Default is [4, 4, 40].
    - soma (bool): Indicates whether the mesh is a soma. Default is False.
    Returns:
    - mesh_dict (dict): A dictionary containing the geometric mesh features.
    """
    mesh_dict = {}

    verts = mesh.vertices
    
    if soma:
        mesh_dict['soma_id'] = mesh_id
        #geometrical features
        mesh_dict['soma_center_mass'] = list(mesh.center_mass)
        mesh_dict['soma_volume_nm'] = mesh.volume
        mesh_dict['soma_area_nm'] = mesh.area
        mesh_dict['soma_area_to_volume'] = area_to_volume(mesh)
        
        radius = np.array([15000 / 4, 15000 / 4, 15000 / 40], dtype=np.int32)
        ctr_pt_vx = ctr_pt_nm  / voxel_resolution
        bbox = np.array([ctr_pt_vx - radius, ctr_pt_vx + radius], dtype=np.int32)
        #MICrONS public flat segmentation 661 does not have a synapse table at that version - query up to date table
        # if mat_version == 661:
        #     syn_df = caveclient.materialize.synapse_query(post_ids=mesh_id, bounding_box=bbox)
        # #Otherwise specify mat_version
        # else:
        syn_df = caveclient.materialize.synapse_query(post_ids=mesh_id, bounding_box=bbox,
                                                    materialization_version=mat_version)
        print('Synapse Table output')
        print(syn_df.shape)
        syn_dict = get_soma_syn_dict(mesh, syn_df)
        mesh_dict.update(syn_dict)
        
    else:
        mesh_dict['nucleus_id'] = mesh_id
        #geometrical features
        mesh_dict['is_watertight'] = mesh.is_watertight
        mesh_dict['nucleus_center_mass'] = list(mesh.center_mass)
        mesh_dict['nucleus_avg_radius'] = avg_radius(mesh)
        mesh_dict['nucleus_volume_nm'] = mesh.volume
        mesh_dict['nucleus_area_nm'] = mesh.area
        mesh_dict['nucleus_area_to_volume_ratio'] = area_to_volume(mesh)
        mesh_dict['nucleus_aspect_ratio'] = aspect_ratio(verts)
        mesh_dict['nucleus_is_clipped'] = is_clipped(mesh)
        mesh_dict = get_fold_features(mesh_id, mesh, 
                      mesh_dict=mesh_dict,threshold = 150)
        
        print("done with %d"%(mesh_id))
    
    return mesh_dict

def get_soma_nuc_features(mesh_dict):
    """
    Updates and returns an input dictionary with features that depend on precomputed
    nucleus and soma features.

    Parameters:
    - mesh_dict (dict): A dictionary containing mesh information.

    Returns:
    - mesh_dict (dict): The updated dictionary with additional features.
    """
    p0 = np.array(mesh_dict['nucleus_center_mass'])
    p1 = np.array(mesh_dict['soma_center_mass'])
    mesh_dict['soma_nuc_d'] = np.linalg.norm(p0 - p1)
    mesh_dict['nucleus_to_soma'] = mesh_dict['nucleus_volume_nm'] / mesh_dict['soma_volume_nm']

    return mesh_dict



def get_fold_features(mesh_id, mesh, 
                      mesh_dict=None,
                      threshold = 150):
    """
    Updates and returns a dictionary with nucleus fold features.

    Parameters:
    - mesh_id (int): The ID of the mesh.
    - mesh (trimesh.base.Trimesh): The input mesh.
    - mesh_dict (dict, optional): The dictionary to update with the fold features. If not provided, a new dictionary will be created.
    - threshold (float, optional): The threshold distance for determining whether a vertex is inside or outside the fold.

    Returns:
    - mesh_dict (dict): The updated dictionary with the following fold features:
        - 'nucleus_id' (int): The ID of the nucleus.
        - 'fold_area_nm' (float): The area of the folded region in nanometers.
        - 'fract_fold' (float): The fraction of the mesh area that is folded.
        - 'avg_fold_depth' (float): The average depth of the folded vertices.

    This function updates the provided dictionary (or creates a new one if not provided) with the fold features of the input mesh. 
    It first performs shrink wrapping on the mesh to create a new mesh. Then, it quantifies the vertices within vs outside the threshold 
    distance to determine the folded region. The fold features include the area of the folded region, 
    the fraction of the mesh area that is folded, and the average depth of the folded vertices.
    """
    new_mesh = shrink_wrap_nucleus(mesh)

    if mesh_dict == None:
        mesh_dict = {'nucleus_id':mesh_id}

    vert_ds, vert_close = new_mesh.kdtree.query(mesh.vertices,k=1)

    folded = np.array([False if v > threshold else True for v in vert_ds])
    fold_mesh = trimesh_io.Mesh(mesh.vertices,
                                mesh.faces,
                                mesh.face_normals,
                                node_mask =folded, 
                                apply_mask=True)
    fold_area = mesh.area - fold_mesh.area

    mesh_dict['fold_area_nm'] =fold_area
    mesh_dict['fract_fold'] = fold_area/mesh.area
    mesh_dict['avg_fold_depth'] = np.mean(vert_ds[vert_ds>threshold])

    return mesh_dict


def get_feat_dict(cell_info, nuc_seg_source, voxel_resolution, fixmesh_input, caveclient=None, mat_version=None, get_nucleus=True, return_mesh=False):
    """
    For a given ID, collects the soma features, nucleus features, and joint features and returns the compiled feature dictionary.
    Parameters:
    - cell_info (tuple): A tuple containing the cell information, including the nucleus ID, soma ID, and center point coordinates.
    - nuc_seg_source (str): The source of the nucleus segmentation.
    - voxel_resolution (float): The voxel resolution.
    - fixmesh_input (str): The input for fixing the mesh.
    - caveclient (optional): The caveclient object. Defaults to None.
    - mat_version (optional): The version of the material. Defaults to None.
    - get_nucleus (bool): Flag indicating whether to get nucleus features. Defaults to True.
    - return_mesh (bool): Flag indicating whether to return the soma mesh. Defaults to False.
    Returns:
    - dict: The compiled feature dictionary.
    Note:
    - This function assumes the existence of the following helper functions: load_fixed_mesh(), get_mesh_features(), load_mesh(), and get_soma_nuc_features().
    """
    
    nuc_id = cell_info[0]
    soma_id = cell_info[1]
    ctr_pt_vx = np.array(cell_info[2])
    ctr_pt_nm = ctr_pt_vx * voxel_resolution
    

    soma_mesh, frac_zero = load_fixed_mesh(soma_id,
                                ctr_pt_nm,
                                fixmesh_input)
    print('fixed soma mesh')

    mesh_dict = get_mesh_features(soma_id, soma_mesh, 
                                  caveclient = caveclient,
                                  mat_version = mat_version,
                                  ctr_pt_nm=ctr_pt_nm,
                                  voxel_resolution=voxel_resolution, 
                                  soma=True)
    mesh_dict['frac_zero'] = frac_zero
    print('got soma mesh dict')
    if get_nucleus == True:
        nuc_mesh = load_mesh(nuc_id, nuc_seg_source,
                             remove_duplicates=True)
        nuc_dict = get_mesh_features(nuc_id, nuc_mesh)
        print('got nuc mesh dict')
        mesh_dict.update(nuc_dict)

        mesh_dict = get_soma_nuc_features(mesh_dict)
    print('cell features collected')
 
    if return_mesh == True:
        del nuc_mesh
        return mesh_dict, soma_mesh
    else:
        del soma_mesh
        del nuc_mesh
        return mesh_dict


    

