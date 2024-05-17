from functools import partial
from taskqueue import LocalTaskQueue
from extract_somatic_features.file_io import get_current_file_ids
from extract_somatic_features.save_features import write_feat_json, write_local_json
import json
import pandas as pd
from multiprocessing import Pool
from functools import partial
from caveclient import CAVEclient



#Input file with all the dataset specific information
inputfile = './extract_somatic_features/minnie_v661/minnie_v661_input.json'
f = open(inputfile)
args = json.load(f)
chunksize = args['chunk']
nucleus_table = args['nucleus_table'] #tablename with nucleus pts for all cells 
client = CAVEclient(args['dataset'])
cv_path = client.info.segmentation_source()

fix_input = {
    'image_source':client.info.image_source(),
    "cv_path" : cv_path,
    'disk_cache_path': args['disk_cache_path'],
    'mip_level':args['mip_level'],
    'cutout_radius':args['cutout_radius']
    }

nuc_lookup = client.materialize.query_view(nucleus_table,
                                            materialization_version=args['materialization_version'])

#filtering errors the have been labeled with pt_root_id 0  
run_df = nuc_lookup.query('pt_root_id != 0') #can be replaced with a df of all the center points to extract features from

#How many cell files already exist
json_filetype = '.json'
json_file_ids = get_current_file_ids(json_filetype,args['output_filepath'],args['gcsfs_project'])
print('There are %d files in Feature JSONs v661'%(len(json_file_ids)))
cells = list(zip(run_df.id, run_df.pt_root_id, run_df.pt_position_lookup))

missing = [i for i in cells if i[1] not in json_file_ids]
print('There are %d missing files'%(len(missing)))

tq = LocalTaskQueue(parallel=1) #modulates parallelization 
tasks = (partial(write_local_json, i,
                nuc_seg_source= args['nuc_seg_source'],
                json_filepath= args['output_filepath'],
                voxel_resolution= args['voxel_resolution'],
                fixmesh_input= fix_input,
                dataset_name = args['dataset']) for i in missing) #only run missing files


tq.insert_all(tasks) # performs on-line execution (naming is historical)

# alterternative serial model
tq.insert(tasks)
tq.execute()

# delete tasks
tq.delete(tasks)
tq.purge() # delete all tasks