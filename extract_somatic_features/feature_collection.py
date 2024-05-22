import numpy as np
from extract_somatic_features.utils import *
from extract_somatic_features.file_io import *


def get_mesh_features(mesh_id, mesh, 
                      caveclient = None,
                      mat_version = None,
                      ctr_pt_nm = [], 
                      voxel_resolution = [4,4,40],
                      soma=False):
    """Returns a dictionary with geometric mesh features. If the mesh is a soma, 
    synapse features will also be added to the dictionary.
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
    """updates and returns an input dictionary with features that depend on precomputed
    nucleus and soma features
    """
    p0 = np.array(mesh_dict['nucleus_center_mass'])
    p1 = np.array(mesh_dict['soma_center_mass'])
    mesh_dict['soma_nuc_d'] = np.linalg.norm(p0 - p1)
    mesh_dict['nucleus_to_soma'] = mesh_dict['nucleus_volume_nm'] / mesh_dict['soma_volume_nm']

    return mesh_dict



def get_fold_features(mesh_id, mesh, 
                      mesh_dict=None,
                      threshold = 150):
    """updates and returns a dictionary with nucleus fold features. This includes shrink wrapping the 
    input mesh and quantifying the vertices within vs outside the threshold distance.
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


def get_feat_dict(cell_info,
                  nuc_seg_source,
                  voxel_resolution,
                  fixmesh_input,
                  caveclient = None,
                  mat_version = None,
                  get_nucleus = True,
                  return_mesh = False):
    """For a given ID, collects the soma features, nucleus features, and joint features and 
    returns the compiled feature dictionary.
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


    

