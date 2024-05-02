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

# example_input={
#     "seg_path" : "https://storage.googleapis.com/neuroglancer/basil_v0/basil_full/seg-aug",
#     "disk_cache_path" : "basil_meshes",
#     "mip_level": 4,
#     "cutout_radius":15
# }

# parallel_example_input={
#     "output_folder": "./meshes/filled_fixed_nuclei",
#     "source_df": "/Users/leilae/Neural_coding/data/NUCLEI.pkl",
#     "id_column":"nucleus_id",
#     "pool_size": 3,
#     "soma_column": None,
# }


def make_linear_transfer(black_val,opaque_val,color=[1,0,0],alpha_max=1.0):
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
    ''' For a given seg ID, gets a bounding box based on the cutout_radius and adjusts the segmentation
    to fill holes and keeps the largest connected component. Returns a mask of this fixed segmentation.'''

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
   
    print(radius_nm,mip, x_radius_pix)
    print('Got bounding boxes')
    #if cutout exceeds volume, resets given axis to volume boundaries   
    mins = np.array(center_vx)-[x_radius_pix,y_radius_pix,z_radius_pix]
    maxs = np.array(center_vx)+[x_radius_pix,y_radius_pix,z_radius_pix]
    print(mins,maxs)
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
    print(bbox)
    seg_cutout = seg_cv.download(
                                bbox,
                                segids=seg_ids,
                                agglomerate=False,
                                mip=mip,
                                coord_resolution=voxel_resolution)
    print('Got seg cutout')
    seg_cutout = np.array(seg_cutout)
    print(seg_cutout.shape)
    seg_cutout = np.squeeze(seg_cutout)
    print('squeezed cutout')
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
    ''' Uses zmesh to create and return a mesh based on the given segmentation mask'''
    
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

def fix_mesh_row(dfrow, fix_mesh, folder, id_column,
                 ctr_pt_column = None,
                 soma_column=None):
    ''' Fixes a mesh with the assumption that the cell information is stored in a row of a dataframe.'''
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
        

    def fix(self, mesh_id, merge_seg_ids = [], ctr_pt_nm = None):
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


    def parallel_fix_from_df(self, df,
                             id_column='nucleus_id',
                             soma_column=None,
                             output_folder='.',
                             ctr_pt_column = None,
                             pool_size=4):
        ''' To fix cells from a dataframe'''

        my_partial = partial(fix_mesh_row,
                             fix_mesh = self,
                             folder = output_folder,
                             id_column=id_column,
                             soma_column = soma_column,
                             ctr_pt_column=ctr_pt_column)
        with Pool(pool_size) as p:
            p.map(my_partial,df.iterrows())
    
    def get_binary_mask(self,mesh_id, 
                        soma_id = None, 
                        ctr_pt_nm = None,
                        output_folder='.',
                        get_og_mask = False):

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

