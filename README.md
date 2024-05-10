# Extract_Somatic_Features
A pipeline to extract soma and nucleus features from the segmentation of precomputed EM datasets. 

This repository is designed to host all the code used to extract nucleus and soma features from precomputed cell segmentations of volumetric electron microscopy data. This implementation was designed to run on the MICrONS dataset but can be adjuested to run on similar datasets. This repository accompanies the manuscript Perisomatic Ultrastructure Efficiently Classifies Cells in Mouse Cortex (Elabbady 2024).

## Installation
The code to run this pipeline has been dockerized as lelabbady/soma_extraction:v16

## Inputs

To run the feature extraction pipeline we use an input JSON file that stores all the dataset and saving specific information. This includes the name of the dataset, necessary links to access the segmentation, and authentication files if saving to the cloud.

Below is the sample input found in sample_input.json. To run this pipeline locally, you can ignore the google and token arguments.

```
{
    "dataset":"minnie65_public_v661",
    "materialization_version": 661,
    "voxel_resolution":[4,4,40],
    "nuc_seg_source" : "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/nuclei",
    "nucleus_table":"nucleus_detection_lookup_v1",
    "output_filepath": "../data",
    "google_bucket": "",
    "gcsfs_project":"",
    "token_file":"",
    "disk_cache_path" : "minnie_meshes_661",
    "mip_level": 3,
    "chunk":10,
    "cutout_radius":15,
    "pool_size":2
}
```

## Running feature extraction locally

To extract the persomatic features from a single cell, run the following script. Note that this will save a single JSON file for each cell with all of the nucleus and soma features. To run feature extraction on a single cell, run the following example script:

```
python ./scripts/run_local_cell.py
```
