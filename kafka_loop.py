import pandas as pa
import numpy as np
import datetime as dt
from run_kafka import run_kafka
import multiprocessing 

def start_kafka(data_folder, state_mask, out_path, start_time, ndays):
    
    run_the_model = run_kafka(data_folder, state_mask, out_path, start_time, ndays)

farm_file = '/data/Sentinel2/uk_farm_keys.csv'

df = pa.read_csv(farm_file)

df = df.iloc[np.where(df['Year!'].iloc[:] == 2018)]

to_use = [n for n,i in enumerate(df['mask_location'].iloc[:]) if type(i) == str]

df = df.iloc[to_use]


active_processes = []

run_on = 9

process_counter = 0

while process_counter <= len(df):
    
    if len(active_processes) <= run_on:
        
        data_dir, mask_name = df.iloc[process_counter][[8, 7]]
        
        start_date = dt.datetime(2018,1,1)
        runs_time = 365
            
        out_dir = '/data/MULTIPLY_outputs/'+mask_name.split('/')[-1].replace('.tif','')
         
        # add it to a process and go away and do it
        p = multiprocessing.Process(target=start_kafka, args = (data_dir, mask_name, out_dir,
                                                               start_date,runs_time))
        
        p.daemon = True
        p.name = str(process_counter)
        
        p.start()
        
        active_processes.append(p)
        
        process_counter += 1
        
    else:
        
        for pr in active_processes:
            
            if pr.is_alive() == False:
                
                active_processes.remove(pr)