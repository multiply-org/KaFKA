#!/usr/bin/env python
import os
from datetime import datetime, timedelta
import numpy as np
import gdal
import osr
import scipy.sparse as sp
import datetime as dt
from shutil import copyfile

# from multiply.inference-engine blah blah blah
try:
    from multiply_prior_engine import PriorEngine
except ImportError:
    pass
import kafka
from kafka.input_output import Sentinel2Observations, KafkaOutput
from kafka import LinearKalman
from kafka.inference.narrowbandSAIL_tools import propagate_LAI_narrowbandSAIL
from kafka.inference import create_prosail_observation_operator
from kafka.inference.narrowbandSAIL_tools import SAILPrior

def reproject_image(source_img, target_img, dstSRSs=None):
    """Reprojects/Warps an image to fit exactly another image.
    Additionally, you can set the destination SRS if you want
    to or if it isn't defined in the source image."""
    g = gdal.Open(target_img)
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterXSize, g.RasterYSize
    xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
    xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
    ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
    ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    xRes, yRes = abs(geo_t[1]), abs(geo_t[5])
    if dstSRSs is None:
        dstSRS = osr.SpatialReference()
        raster_wkt = g.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)
    else:
        dstSRS = dstSRSs
    g = gdal.Warp('', source_img, format='MEM',
                  outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                  dstSRS=dstSRS)
    return g

class KafkaOutputMemory(object):
    """A very simple class to output the state."""
    def __init__(self, parameter_list):
        self.parameter_list = parameter_list
        self.output = {}
    def dump_data(self, timestep, x_analysis, P_analysis, P_analysis_inv,
                state_mask):
        solution = {}
        for ii, param in enumerate(self.parameter_list):
            solution[param] = x_analysis[ii::7]
        self.output[timestep] = solution

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

if __name__ == "__main__":    
    
    #####################################################################
    #////////////////////////////////////////////////////////////////////
    
    Qfactor = 0.05
    propagator = None#propagate_LAI_narrowbandSAIL   
    
    location = 'Quzou'
    run_time = dt.datetime.now().strftime('%Y%m%d_%H%M')
    
    # Format of sband strong:
    # 'NUM,NUM,NUM', padded to 2 numbers with 0.
    # Use 8A, NOT 9. 
    band_string = '02,03,04,08,11,12'
    
    input_prior = False
    
    if propagator != None:
       prop_flag = 'T'
       q_flag = Qfactor
    else:
       prop_flag = 'F'
       q_flag = ''
       
    if input_prior != None:
       prior_flag = 'T'
    else:
       prior_flag = 'F'
    
    runhandle = 'S2_%s_%s_Pri%s_Pro%s%s'%(location,run_time,prior_flag,prop_flag,q_flag)
    
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    #####################################################################
    # Set up logging
    import logging
    logging.basicConfig( 
    level=logging.getLevelName(logging.DEBUG), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="logfiles/%s.log"%runhandle)

    # Set up logging
    Log = logging.getLogger(__name__+".kafka_test_x.py")

    #runname = 'tstep_bug'  #Used in output directory as a unique identifier
    # To run without propagation set propagator to None and set a
    # prior in LinearKalman.
    # If propagator is set to propagate_LAI_narrowbandSAIL then the
    # prior in LinearKalman must be set to None because this propagator
    # includes a prior

    parameter_list = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm',
                      'lai', 'ala', 'bsoil', 'psoil']

    Log.info("propagator = {}".format(propagator))
    
    
    # split the band string so it can iterated through
    band_string = band_string.split(',')
    
    #path = "/home/acornelius/KaFKA-InferenceEngine/outputs/sensitivity/runWith_%s"%('_'.join(band_string).replace("'",''))
    #if not os.path.exists(path):
    #    mkdir_p(path)

    # create a reop for the results
    path = "/home/acornelius/KaFKA-InferenceEngine/outputs/%s"%runhandle
    if not os.path.exists(path):
        mkdir_p(path)

    #emulator_folder = "/home/ucfafyi/DATA/Multiply/emus/sail/"
    emulator_folder = "/home/glopez/Multiply/src/py36/emus/sail"
    #emulator_folder = "/data/archive/emulators/s2_prosail"

    ## Barrax
    #state_mask = "./Barrax_pivots.tif"
    #data_folder = "/data/001_planet_sentinel_study/sentinel/30/S/WJ"
    #start_time = "2017001"
    #time_grid_start = datetime(2017, 1, 1)
    #num_days = 366

    ## California
    start_time = "2017001"
    time_grid_start = datetime(2017, 1, 1)
    #time_grid_start = datetime(2017, 12, 31)
    data_folder = "/home/acornelius/sent/cloud_free"
    #state_mask = "/data/001_planet_sentinel_study/planet/utm11n_sur_ref/field_sites.tif"
    #state_mask = "/home/npounder/repositories/python3/KaFKA-InferenceEngine/dataCleansing/Window200Pix.tif"
    state_mask = '/home/acornelius/workspace/quzou_40by40_state_mask_NOTHING_HERE.tif'
    num_days = 365

    Log.info("start_time = {}".format(start_time))
    
    # copy relivant files accross for records:
    copyfile("/home/acornelius/KaFKA-InferenceEngine/local_kafka_test_S2_version_alex.py",path+'/meta_script.py')
    if input_prior == True:
        copyfile("/home/acornelius/KaFKA-InferenceEngine/kafka/inference/narrowbandSAIL_tools.py",path+'/prior_settings.py')

    s2_observations = Sentinel2Observations(data_folder,
                                            emulator_folder,
                                            state_mask,
                                            band_string)    

    projection, geotransform = s2_observations.define_output()

    output = KafkaOutput(parameter_list, geotransform,
                         projection, path)

    # If using a separate prior then this is passed to LinearKalman
    # Otherwise this is just used to set the starting state vector.
    the_prior = SAILPrior(parameter_list, state_mask)
    g = gdal.Open(state_mask)
    mask = g.ReadAsArray().astype(np.bool)
    
    #raise ValueError('here')
    
    # prior = None when using propagate_LAI_narrowbandSAIL
    kf = LinearKalman(s2_observations, output, mask,
                      create_prosail_observation_operator,
                      parameter_list,
                      state_propagation=propagator,
                      prior=the_prior,
                      linear=False)

    # Check if there's a prior from a previous timestep
    P_inv_fname = "P_analysis_inv_%s.npz" % time_grid_start.strftime("A%Y%j")
    P_inv_fname = os.path.join( path, P_inv_fname )

    x_fname = "x_analysis_%s.npz" % time_grid_start.strftime("A%Y%j")
    x_fname = os.path.join(path, x_fname)

    if os.path.exists(P_inv_fname) and os.path.exists(x_fname):
        # Load stored matrices...
        x_forecast = np.load(x_fname)['arr_0']
        P_forecast_inv = sp.load_npz(P_inv_fname)
    else:
        # Get starting state... We can request the prior object for this
        x_forecast, P_forecast_inv = the_prior.process_prior(None)

    # Inflation amount for propagation
    Q = np.zeros_like(x_forecast)
    Q[6::10] = Qfactor

    kf.set_trajectory_model()
    kf.set_trajectory_uncertainty(Q)

    # This determines the time grid of the retrieved state parameters
    time_grid = list((time_grid_start + timedelta(days=x)
                     for x in range(0, num_days, 1)))
    kf.run(time_grid, x_forecast, None, P_forecast_inv,
           iter_obs_op=True)

