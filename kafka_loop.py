import pandas as pa
import numpy as np
import datetime as dt
from run_kafka_trajectory import run_kafka_trajectory
from run_kafka import run_kafka
import multiprocessing 
from kafka.inference.narrowbandSAIL_tools import sail_prior_values
import glob
import gdal
from kafka_prior import kafka_prior
from shutil import copyfile
import os
import copy

# >>>>>>>>>>>>>> USER INPUT
# Enter the location of the farm file here:
farm_file = "/home/acornelius/KaFKA-InferenceEngine/new_version/KaFKA/2018_farm_key.csv"

# enter some identifier for the runs
run_handle = 'uk_all2018_MoLAIs'

root = '/data/MULTIPLY_outputs/'

use_trajectory = True

Qfactor = 1

year = 2018
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def start_kafka(data_folder, state_mask, out_path, start_time, ndays):
    
    run_the_model = run_kafka(data_folder, state_mask, out_path, start_time, ndays)

def start_kafka_traj(data_folder, state_mask, out_path, start_time, ndays, prior, Qfactor):
    
    run_the_model = run_kafka_trajectory(data_folder, state_mask, out_path, start_time, ndays, prior,Qfactor)


df = pa.read_csv(farm_file)

df = df.iloc[np.where(df['Year!'].iloc[:] == year)]

to_use = [n for n,i in enumerate(df['mask_location'].iloc[:]) if type(i) == str]

df = df.iloc[to_use]

prior_compile = kafka_prior(df, do_smooth=True)
prior_compile.get_modis_lai()

profiles = prior_compile.output()

mean, covar, inv_covar = sail_prior_values()

lai_prior = mean[6]
lai_prior = np.log(lai_prior)*-2
lai_sigma = covar[6,6]

now = dt.datetime.now()
now_str = '%s%02d%02d_%s%s'%(now.year,now.month,now.day,now.hour,now.minute)

if run_handle == None:
    run_handle = 'kafka'

dir_name = '%s_%sp_%ssig_%s/'%(run_handle,now_str,np.round(lai_prior,3),np.round(lai_sigma,3))

failed_clicker = 1

active_processes = []

run_on = 8

process_counter = 0

while process_counter <= len(df):
    
    if len(active_processes) < run_on:
        
        data_dir, mask_name = df.iloc[process_counter][[6, 7]]

        start_date = dt.datetime(2018,1,2)
        runs_time = 365

        out_dir = root+dir_name+mask_name.split('/')[-1].replace('.tif','')

        try:

            if use_trajectory == False:
                p = multiprocessing.Process(target=start_kafka, args = (data_dir, mask_name, out_dir,
                                                               start_date,runs_time))

            else:

                p = multiprocessing.Process(target=start_kafka_traj, 
                                            args=(data_dir, mask_name, out_dir,\
                                               start_date,runs_time,profiles[process_counter],\
                                               Qfactor))

            p.daemon = True
            p.name = str(process_counter)

            p.start()

            active_processes.append(p)

            process_counter += 1

        except:
            print ('Failed on step ', process_counter)

            if failed_clicker == 1:

                failed_df = copy.deepcopy(df.iloc[process_counter:process_counter+1])

                failed_clicker = 0

            else:

                failed_df.append(df.iloc[process_counter])
                                            
    else:
        
        for pr in active_processes:
            
            if pr.is_alive() == False:
                
                active_processes.remove(pr)
                                            

copyfile(os.path.abspath(__file__), root+dir_name+'run_file.py')

if failed_clicker == 0:
    failed_df.to_csv(root+dir_name+'farms_that_failed.csv')
                                            
# import pandas as pa
# import numpy as np
# import datetime as dt
# from run_kafka import run_kafka
# import multiprocessing 

# def start_kafka(data_folder, state_mask, out_path, start_time, ndays):
    
#     run_the_model = run_kafka(data_folder, state_mask, out_path, start_time, ndays)

# farm_file = '/data/Sentinel2/uk_farm_keys.csv'

# df = pa.read_csv(farm_file)

# df = df.iloc[np.where(df['Year!'].iloc[:] == 2018)]

# to_use = [n for n,i in enumerate(df['mask_location'].iloc[:]) if type(i) == str]

# df = df.iloc[to_use]


# active_processes = []

# run_on = 9

# process_counter = 0

# while process_counter <= len(df):
    
#     if len(active_processes) <= run_on:
        
#         data_dir, mask_name = df.iloc[process_counter][[8, 7]]
        
#         start_date = dt.datetime(2018,1,1)
#         runs_time = 365
            
#         out_dir = '/data/MULTIPLY_outputs/'+mask_name.split('/')[-1].replace('.tif','')
         
#         # add it to a process and go away and do it
#         p = multiprocessing.Process(target=start_kafka, args = (data_dir, mask_name, out_dir,
#                                                                start_date,runs_time))
        
#         p.daemon = True
#         p.name = str(process_counter)
        
#         p.start()
        
#         active_processes.append(p)
        
#         process_counter += 1
        
#     else:
        
#         for pr in active_processes:
            
#             if pr.is_alive() == False:
                
#                 active_processes.remove(pr)