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

### Parallelization

To parallelize the jobs we use TaskQueue and Amazon SQS. An example script can be found below. Note that you will need to set up  your own queue to feed in to the task worker command:
```
python ./scripts/run_local_tq.py
```

## Running feature extraction on kubernetes

To run feature extraction on the entire MICrONS dataset, we used multiple nodes on kubernetes. Running all the jobs inside separate docker containers. The script for this setup is under:
```
./scripts/soma_nuc_task.yaml
```
Note that pipeline specific arguments like DOCKER_CONTAINER and SQS_QUEUE are written in all caps and will need to be changed accordingly.

## Other resources
### To see how these features have been used, please check out the accompanying manuscript:
- Perisomatic Ultrastructure Efficiently Classifies Cells in Mouse Cortex [(Elabbady 2022)](https://www.biorxiv.org/content/10.1101/2022.07.20.499976v2)

### To see how these features have enabled scientific discovery, please check out the following papers:
- Cell-type-specific inhibitory circuitry from a connectomic census of mouse visual cortex [(Schneider-Mizell 2023)](https://biorxiv.org/content/10.1101/2023.01.23.525290v3)
- The Synaptic Architecture of Layer 5 Thick Tufted Excitatory Neurons in the Visual Cortex of Mice [(Bodor 2023)](https://www.biorxiv.org/content/10.1101/2023.10.18.562531v1)
- Integrating EM and Patch-seq data: Synaptic connectivity and target specificity of predicted Sst transcriptomic types [(Gamlin 2023)](https://www.biorxiv.org/content/10.1101/2023.03.22.533857v1)
