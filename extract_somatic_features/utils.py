import vtk
import trimesh
import numpy as np
import urllib.request, json
from meshparty import trimesh_io, trimesh_vtk, mesh_filters
from sklearn.decomposition import PCA
from scipy import sparse
from scipy.ndimage import morphology 


def grab_seg_dict(neuroglancer_url, layer='seg-aug'):
    """
    Retrieves the segmentation dictionary from a Neuroglancer URL.
    Parameters:
    - neuroglancer_url (str): The Neuroglancer URL containing the JSON data.
    - layer (str): The layer name to extract the segmentation dictionary from. Default is 'seg-aug'.
    Returns:
    - layer_dict (dict): The segmentation dictionary extracted from the specified layer.
    """
 
    json_url = neuroglancer_url.split('json_url=')[1]
    print(json_url)

    response = urllib.request.urlopen(json_url)
    data = json.loads(response.read())
    layer_dict = data['layers'][layer]
    layer_dict['num_segments'] = len(layer_dict['segments'])
    print(layer_dict['num_segments'])
    return layer_dict


def avg_radius(mesh):
    """
    Calculate the average radius of a mesh.
    Parameters:
    - mesh: A mesh object representing the 3D geometry.
    Returns:
    - The average radius of the mesh.
    """
    v = mesh.vertices
    
    return np.mean(np.linalg.norm(v - mesh.center_mass, axis=1))

def area_to_volume(mesh):
    """
    Calculates the ratio of surface area to volume for a given mesh.

    Parameters:
    - mesh: The mesh object for which to calculate the ratio.

    Returns:
    - The ratio of surface area to volume.
    """
    v = mesh.volume
    a_over_v = mesh.area/v
    radius = np.cbrt((v/((4/3)*np.pi)))
    return (a_over_v)*(radius/3)

def calc_bb(vertices):
    """
    Calculates the bounding box corner coordinates for a given set of vertices.

    Parameters:
    vertices (numpy.ndarray): An array of shape (N, 3) representing the vertices.

    Returns:
    numpy.ndarray: An array of shape (8, 3) containing the bounding box corner coordinates.

    """
    aligned_axes = np.array([[min(vertices[:,0]),min(vertices[:,1]),min(vertices[:,2])],
                             [max(vertices[:,0]),max(vertices[:,1]),max(vertices[:,2])]]) 
    bb = trimesh.bounds.corners(aligned_axes)

    return bb

def remove_small_components(mesh, size_thresh=200):
    """
    This function takes in a mesh and removes components smaller than a given threshold. 

    Parameters:
    - mesh: The input mesh.
    - size_thresh: The threshold size for removing components. Default is 200.

    Returns:
    - new_mesh: The new mesh with small components removed.
    """
    mesh_filters.filter_components_by_size(mesh, min_size=size_thresh)
    return mesh.apply_mask(mesh)


def get_largest_components(mesh):
    """
    This function is meant to remove small segmentations within the larger mesh and clean up the mesh data.
    Parameters:
    - mesh: The input mesh.
    Returns:
    - new_mesh: The new mesh with only the largest connected component.
    """
    is_big = mesh_filters.filter_largest_component(mesh)
    return mesh.apply_mask(is_big)

def is_clipped(mesh, buffer=3000):
    """
    Check if a mesh is clipped based on its bounding box coordinates.
    Parameters:
    - mesh: The mesh object to check.
    - buffer: The buffer value to add to the bounding box coordinates.
    Returns:
    - clipped: A boolean value indicating whether the mesh is clipped or not.
    """

    clipped = False
    v = mesh.vertices
    bb = calc_bb(v)
    
    if (min(bb[:,0]) <= (100000 + buffer) or max(bb[:,0]) >= (660000 - buffer)):
        clipped = True
    elif (min(bb[:,1]) <= (80000+ buffer) or max(bb[:,1]) >= (1000000 - buffer)):
        clipped = True
    elif (min(bb[:,2]) <= (5600+ buffer) or max(bb[:,2]) >= (37600 - buffer)):
        clipped = True
    
    return clipped
    

def surrounding_soma(mesh, mesh_id, cv, cutout_size=10):
    """
    Parameters:
    - mesh (ndarray): The mesh data.
    - mesh_id (int): The ID of the mesh.
    - cv (ndarray): The cutout volume.
    - cutout_size (int): The size of the cutout volume.
    Returns:
    - int: The most common value surrounding the specified mesh ID - assigned soma ID
    """

    mesh_cm_vx = np.floor(mesh.center_mass/cv.resolution)

    cutout_vol = cv[int(mesh_cm_vx[0]-cutout_size):mesh_cm_vx[0]+cutout_size,
                    mesh_cm_vx[1]-cutout_size:mesh_cm_vx[1]+cutout_size,
                    mesh_cm_vx[2]-cutout_size:mesh_cm_vx[2]+cutout_size]
                    
    mask = cutout_vol == mesh_id
    n = cutout_vol[morphology.binary_dilation(mask)]
    n = n[n!= mesh_id]
    vals, counts = np.unique(n, return_counts=True)
    return vals[np.argmax(counts)]

def principal_orientation(vertices, up):
    """
    Calculate the principal orientation of a set of vertices. 

    Parameters:
    - vertices (array-like): The vertices to calculate the principal orientation for.
    - up (array-like): The up vector to compare the principal orientation against.

    Returns:
    - float: The absolute dot product between the unit first component of the principal components and the up vector.
    """

    pca = PCA(n_components=3)
    pca.fit(vertices)
    components = pca.components_
    unit_first = components[0]/np.linalg.norm(components[0])
    return np.absolute(np.dot(unit_first,up))

def aspect_ratio(vertices):
    """
    Calculate the aspect ratio of a set of vertices as a measure of how spherical
    the object is. 1 = perfect sphere, 1> more elliptical.

    Parameters:
    - vertices (array-like): The vertices of the object.

    Returns:
    - float: The aspect ratio of the object.
    """
    pca = PCA(n_components=3)
    pca.fit(vertices)
    variance_v = pca.explained_variance_
    return variance_v[0]/(sum(variance_v[1:])/2)

def avg_nuc_to_soma_dist(soma_mesh, nuc_mesh):
    """
    Calculate the average and standard deviation of the distance between the soma and each nucleus.

    Parameters:
    - soma_mesh (Mesh): The mesh representing the soma.
    - nuc_mesh (Mesh): The mesh representing the nuclei.

    Returns:
    - tuple: A tuple containing the average and standard deviation of the distances.
    """
    nv = nuc_mesh.vertices[::100,:]
    nearest = soma_mesh.kdtree.query(nv)
    return np.mean(nearest[0]), np.std(nearest[0])

def shrink_wrap_nucleus(mesh, initial_radius=12000, initial_theta_resolution=256, n_iters=2, max_feature_size=1000, final_detail=300):
    """
    Shrink wraps a given mesh to a nucleus shape.
    Args:
        mesh (trimesh_io.Mesh): The input mesh to be shrink wrapped.
        initial_radius (float): The initial radius of the nucleus shape. Default is 12000.
        initial_theta_resolution (int): The initial theta resolution of the nucleus shape. Default is 256.
        n_iters (int): The number of iterations for the shrink wrapping process. Default is 2.
        max_feature_size (int): The maximum feature size for subdivision during the shrink wrapping process. Default is 1000.
        final_detail (int): The final detail of the shrink wrapped mesh. Default is 300.
    Returns:
        trimesh_io.Mesh: The shrink wrapped mesh.
    """

    mesh.vertices = mesh.vertices-np.mean(mesh.vertices, axis=0)
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(0,0,0)
    sphereSource.SetRadius(initial_radius)
    sphereSource.SetThetaResolution(initial_theta_resolution)
    sphereSource.Update()
    spherepoly = sphereSource.GetOutput()
    
    
    verts, faces, other = trimesh_vtk.poly_to_mesh_components(spherepoly)
    sphere_mesh = trimesh_io.Mesh(verts, faces)
    print(sphere_mesh.is_watertight)
    ds, close = mesh.kdtree.query(verts, k=1)
    new_mesh = trimesh_io.Mesh(mesh.vertices[close,:], faces, process=False)
    print(new_mesh.is_watertight)
    for i in range(n_iters):
        print(new_mesh.vertices.shape)
        print('after moving',new_mesh.is_watertight)
        n_shrink_verts, n_shrink_faces = trimesh.remesh.subdivide_to_size(new_mesh.vertices,
                                                                          new_mesh.faces,
                                                                          max_feature_size)
        new_mesh = trimesh_io.Mesh(n_shrink_verts, n_shrink_faces, process=False)
        print('after subdivide', new_mesh.is_watertight)
        ds, close = mesh.kdtree.query(new_mesh.vertices,k=1)
        new_mesh.vertices = mesh.vertices[close,:]
    n_shrink_verts, n_shrink_faces = trimesh.remesh.subdivide_to_size(new_mesh.vertices,
                                                                      new_mesh.faces,
                                                                      final_detail)
    new_mesh = trimesh_io.Mesh(n_shrink_verts, n_shrink_faces)
    return new_mesh


def get_soma_syn_dict(mesh, syn_df):
    syn_dict = {}
    if len(syn_df) > 0:
        syn_pos = np.vstack(syn_df.ctr_pt_position) * [4, 4, 40]
        ds, syn_soma_pt = mesh.kdtree.query(syn_pos, k=1)
        soma_syn_df = syn_df[ds < 150]
        syn_dict["n_soma_syn"] = int(soma_syn_df.shape[0])
        syn_dict["soma_syn_density"] = soma_syn_df.shape[0] / mesh.area
    else:
        syn_dict["n_soma_syn"] = 0
        syn_dict["soma_syn_density"] = 0

    return syn_dict



