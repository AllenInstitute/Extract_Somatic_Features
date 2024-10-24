import numpy as np
from scipy import ndimage
from meshparty import trimesh_io, mesh_filters
from caveclient import CAVEclient
import cloudvolume as cv
import imageryclient as ic
import pandas as pd
from multiprocessing import Pool
from functools import partial
import os
import tifffile as tif
import zmesh



def make_linear_transfer(black_val, opaque_val, color=[1,0,0], alpha_max=1.0):
    """
    Creates a linear transfer function for volume rendering.

    Parameters:
    - black_val (float): The value at which the transfer function starts to become opaque.
    - opaque_val (float): The value at which the transfer function becomes fully opaque.
    - color (list, optional): The color of the transfer function. Defaults to [1, 0, 0] (red).
    - alpha_max (float, optional): The maximum opacity value. Defaults to 1.0.

    Returns:
    - tf (list): The linear transfer function as a list of control points, where each control point is a list of [value, red, green, blue, alpha].
    """
    tf = []
    tf.append([-40,color[0],color[1],color[2],0])
    tf.append([black_val,color[0],color[1],color[2],0])
    tf.append([opaque_val,color[0],color[1],color[2],alpha_max])
    tf.append([300,color[0],color[1],color[2],alpha_max])
    return tf


def get_fixed_seg_mask(seg_id,
                       cvpath,
                       cache_path,
                       mip,
                       imageclient,
                       cutout_radius,
                       voxel_resolution=[4,4,40],
                       merge_seg_ids = [],
                       ctr_pt_nm = None,
                       vertical_dilation_pix = 5,
                       get_og_mask = False):

    """
    Retrieves the fixed segmentation mask for a given segment ID.
    Args:
        seg_id (int): The segment ID.
        cvpath (str): The path to the CloudVolume.
        cache_path (str): The path to the disk cache.
        mip (int): The desired resolution level.
        imageclient: The image client object.
        cutout_radius (float): The radius of the cutout in micrometers.
        voxel_resolution (list, optional): The voxel resolution in nanometers. Defaults to [4,4,40].
        merge_seg_ids (list, optional): Additional segment IDs to merge. Defaults to [].
        ctr_pt_nm (numpy.ndarray, optional): The center point in nanometers. Defaults to None.
        vertical_dilation_pix (int, optional): The number of pixels for vertical dilation. Defaults to 5.
        get_og_mask (bool, optional): Whether to get the original mask. Defaults to False.
    Returns:
        tuple: A tuple containing the fixed segmentation mask, the center point in nanometers, the radius in nanometers, and the fraction of zero values in the mask.
    """

    print('Getting Fixed Mesh')
    if ctr_pt_nm is None:
        mm = trimesh_io.MeshMeta(cv_path=cvpath, disk_cache_path=cache_path)
        mesh = mm.mesh(seg_id=seg_id)

        center_nm = np.mean(mesh.vertices, axis=0)
    else:
        center_nm = ctr_pt_nm
    
    center_vx = ctr_pt_nm / voxel_resolution
    radius_nm = cutout_radius*1000
    
    
    x_radius_pix = int((radius_nm/voxel_resolution[0]))
    y_radius_pix = int((radius_nm/voxel_resolution[1]))
    z_radius_pix = int(radius_nm/voxel_resolution[2])
   
    #print(radius_nm,mip, x_radius_pix)
    print('Got bounding boxes')
    #if cutout exceeds volume, resets given axis to volume boundaries   
    mins = np.array(center_vx)-[x_radius_pix,y_radius_pix,z_radius_pix]
    maxs = np.array(center_vx)+[x_radius_pix,y_radius_pix,z_radius_pix]
    #print(mins,maxs)
    seg_ids = [seg_id]
    if len(merge_seg_ids) > 0:
        seg_ids += merge_seg_ids
    
    seg_cv = cv.CloudVolume(
                cvpath,
                use_https=True,
                fill_missing=True,
                bounded=False,
                progress=False
            )
    print('created cv object')
    bbox = cv.Bbox(mins, maxs,dtype=np.int32)
    #print(bbox)
    seg_cutout = seg_cv.download(
                                bbox,
                                segids=seg_ids,
                                agglomerate=False,
                                mip=mip,
                                coord_resolution=voxel_resolution)
    print('Got seg cutout')
    seg_cutout = np.array(seg_cutout)
    #print(seg_cutout.shape)
    seg_cutout = np.squeeze(seg_cutout)
    #print('squeezed cutout')
    seg_cutout = np.squeeze(seg_cutout)
    print('squeezed cutout')
    frac_zero = np.count_nonzero(seg_cutout==0)/seg_cutout.size

    #merging nucleus and soma id if cell, otherwise taking only nucleus id
    og_mask = np.isin(seg_cutout, seg_ids)
    print('ran isin')
    og_mask = np.squeeze(og_mask)
    print('got mask')
    print(True in og_mask)
    
    #vertical dilation to fill missing slits in segmentation
    f = np.zeros((vertical_dilation_pix,vertical_dilation_pix,vertical_dilation_pix), bool)
    f[int(vertical_dilation_pix/2),int(vertical_dilation_pix/2):]=True
    dilated_mask = ndimage.binary_dilation(og_mask, f)
    zeros = np.logical_and(dilated_mask, seg_cutout ==0)
    merged = np.logical_or(og_mask,zeros)

    #removing internal and external segmentation junk, filling holes
    st_elem=ndimage.generate_binary_structure(3,1)
    seg_mask = ndimage.binary_dilation(merged, st_elem)
    
    seg_mask = ndimage.binary_fill_holes(seg_mask)
    seg_mask = ndimage.binary_erosion(seg_mask, st_elem)
    seg_mask = ndimage.binary_erosion(seg_mask, st_elem)
    mask = seg_mask.astype(np.uint8)

    return mask, center_nm, radius_nm, frac_zero


def get_trimesh_from_segmask(seg_mask, og_cm, cutout_radius, mip, imageclient):
    """
    Generate a trimesh object from a segmentation mask.
    Parameters:
    - seg_mask (ndarray): The segmentation mask.
    - og_cm (ndarray): The original center of mass.
    - cutout_radius (float): The radius of the cutout.
    - mip (int): The MIP level.
    - imageclient (ImageClient): The image client.
    Returns:
    - new_mesh (trimesh.Mesh): The generated trimesh object.
    """
    
    mip_resolution = imageclient.segmentation_cv.mip_resolution(mip)
    print('THIS IS YOUR MIP RESOLUTION: ' + str(mip_resolution))
    spacing_nm = mip_resolution
    print(seg_mask.shape, spacing_nm, og_cm-cutout_radius)
    mesher = zmesh.Mesher( spacing_nm ) # anisotropy of image
    mesher.mesh(seg_mask.T) # initial marching cubes pass
    mesh = mesher.get_mesh(1, normals=False,
                            simplification_factor=100, 
                             max_simplification_error=8)
    mesher.erase(1) # delete high res mesh
    mesher.clear() # clear memory retained by mesher

    origin=og_cm-cutout_radius

    new_mesh = trimesh_io.Mesh(mesh.vertices + origin, mesh.faces)
    print('got new mesh')
    print(new_mesh.vertices.shape)
    is_big = mesh_filters.filter_largest_component(new_mesh)
    new_mesh = new_mesh.apply_mask(is_big)
    
    return new_mesh


def fix_mesh_row(dfrow, fix_mesh, folder, id_column, ctr_pt_column=None, soma_column=None):
    """
    Fix the mesh for a given row in a DataFrame and saves the fixed mesh as an h5 file.
    Parameters:
    - dfrow (tuple): A tuple containing the index and the row of the DataFrame.
    - fix_mesh (object): An object representing the mesh fixer.
    - folder (str): The path to the folder containing the mesh files.
    - id_column (str): The name of the column containing the mesh IDs.
    - ctr_pt_column (str, optional): The name of the column containing the center point coordinates. Defaults to None.
    - soma_column (str, optional): The name of the column containing the soma IDs. Defaults to None.
    Returns:
    - None
    Raises:
    - None
    """
    (ind,dfrow)=dfrow
    mesh_id = dfrow[id_column]
    obj_path = os.path.join(folder, "%s.obj"%mesh_id)
    h5_path = os.path.join(folder, "%s.h5"%mesh_id)
    
    if soma_column is not None:    
        seg_ids = dfrow[soma_column]
        print(len(seg_ids))
    else:
        seg_ids = []    
    if ctr_pt_column is not None:
        ctr_pt = np.array(dfrow[ctr_pt_column]) 
    else:
        ctr_pt = None
    new_mesh = fix_mesh.fix(mesh_id, merge_seg_ids=seg_ids, ctr_pt_nm=ctr_pt)
    print('This is your new center_mass: ' + str(new_mesh.center_mass))
    trimesh_io.write_mesh_h5(h5_path, new_mesh.vertices, new_mesh.faces, overwrite=True)
    print('mesh saved')


class FixMesh():
    
    def __init__(self,image_source,dataset_name=None, dynamic_seg=True, disk_cache_path=None,
                      mip_level=4,  cutout_radius=15, cv_path = None, resolution=[4,4,40]):

        self.disk_cache_path = disk_cache_path
        self.mip_level = mip_level
        self.cutout_radius = cutout_radius
        self.dataset = dataset_name
        self.resolution = resolution

        if cv_path:
            self.imageclient = ic.ImageryClient(segmentation_source=cv_path,
                                      image_source=image_source,
                                      resolution=resolution)
            self.seg_path = cv_path
            

        else:
            client = CAVEclient(self.dataset)
            self.infoclient = client.info.InfoServiceClient(dataset_name=self.dataset)
            self.seg_path = client.info.segmentation_source()
        

    def fix(self, mesh_id, merge_seg_ids=[], ctr_pt_nm=None):
        """
        Fixes the given mesh by applying segmentation mask and returns the fixed mesh.

        Parameters:
        - mesh_id (str): The ID of the mesh to be fixed.
        - merge_seg_ids (list, optional): List of segment IDs to be merged. Default is an empty list.
        - ctr_pt_nm (None or tuple, optional): Center point coordinates in nanometers. Default is None.

        Returns:
        - new_mesh (trimesh.Trimesh): The fixed mesh.
        - frac_zero (float): Fraction of zero values in the segmentation mask.
        """
        seg_mask, og_cm, radius_nm, frac_zero = get_fixed_seg_mask(mesh_id,
                                                        self.seg_path,
                                                        self.disk_cache_path,
                                                        self.mip_level,
                                                        self.imageclient,
                                                        self.cutout_radius,
                                                        voxel_resolution = self.resolution,
                                                        merge_seg_ids=merge_seg_ids,
                                                        ctr_pt_nm=ctr_pt_nm)  
        print(np.sum(seg_mask))
        new_mesh = get_trimesh_from_segmask(seg_mask, og_cm, radius_nm,self.mip_level,
                                            self.imageclient)
        return new_mesh, frac_zero


    def parallel_fix_from_df(self, df, id_column='nucleus_id', soma_column=None, output_folder='.', ctr_pt_column=None, pool_size=4):
        """
        Perform parallel fixing of mesh based on the given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            id_column (str, optional): The column name for the nucleus ID. Defaults to 'nucleus_id'.
            soma_column (str, optional): The column name for the soma data. Defaults to None.
            output_folder (str, optional): The output folder path. Defaults to '.'.
            ctr_pt_column (str, optional): The column name for the center point data. Defaults to None.
            pool_size (int, optional): The number of processes to use for parallelization. Defaults to 4.
        """

        my_partial = partial(fix_mesh_row,
                             fix_mesh = self,
                             folder = output_folder,
                             id_column=id_column,
                             soma_column = soma_column,
                             ctr_pt_column=ctr_pt_column)
        with Pool(pool_size) as p:
            p.map(my_partial,df.iterrows())
    
    def get_binary_mask(self, mesh_id, soma_id=None, ctr_pt_nm=None, output_folder='.', get_og_mask=False):
        """
        Generates a binary mask for a given mesh ID.
        Parameters:
            mesh_id (int): The ID of the mesh.
            soma_id (int, optional): The ID of the soma. Defaults to None.
            ctr_pt_nm (float, optional): The center point in nanometers. Defaults to None.
            output_folder (str, optional): The output folder path. Defaults to '.'.
            get_og_mask (bool, optional): Flag to get the original mask. Defaults to False.
        Returns:
            numpy.ndarray: The generated binary mask.
        Raises:
            None
        """

        filename = '%d_binarymask_mip%d_cutout%d.tiff'%(mesh_id,self.mip_level,self.cutout_radius)
        filename = os.path.join(output_folder,filename)

        if not os.path.isfile(filename) and mesh_id != 0:

            mask, og_cm, radius_nm = get_fixed_seg_mask(mesh_id,
                                                        self.seg_path,
                                                        self.disk_cache_path,
                                                        self.mip_level,
                                                        self.imageclient,
                                                        self.cutout_radius,
                                                        cell_seg_id=soma_id,
                                                        ctr_pt_nm=ctr_pt_nm,
                                                        get_og_mask=get_og_mask)  
                                        
            
            with tif.TiffWriter(filename, bigtiff=True) as tifW:
                for i in range(mask.shape[0]):
                    tifW.save(mask[i])
            return mask
        else:
            print(str(mesh_id) + " exists")

