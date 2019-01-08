#!/usr/bin/env python

import logging
logging.basicConfig( 
    level=logging.getLevelName(logging.DEBUG), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="logfiles/Planet_goodmask__nonan_Q0-05_p4_e1.log")

import os
from datetime import datetime, timedelta
import numpy as np


import numpy as np

import gdal

import osr

import scipy.sparse as sp

# from multiply.inference-engine blah blah blah
try:
    from multiply_prior_engine import PriorEngine
except ImportError:
    pass


import kafka
from kafka.input_output import KafkaOutput
from kafka.input_output.Planetlabs_Observations import PlanetlabsObservations
from kafka import LinearKalman
from kafka.inference.narrowbandSAIL_tools import propagate_LAI_narrowbandSAIL
from kafka.inference import create_prosail_observation_operator
from kafka.inference.narrowbandSAIL_tools import SAILPrior
# Probably should be imported from somewhere else, but I can't see
# where from ATM... No biggy


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

def mkdir_p(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as exc:
        if exc.errno == Q0errno.EEXIST and os.path.isdir(dirpath):
            pass
        else:
            raise

if __name__ == "__main__":

    # Set up logging
    Log = logging.getLogger(__name__+".kafka_test_x.py")


    runname = 'planet_goodmask_nonan_Q0-05_p4_e1'  #Used in output directory as a unique identifier
    #runname = 'planet_multipleobsbug'  #Used in output directory as a unique identifier
    Qfactor = 0.05
    # To run without propagation set propagator to None and set a
    # prior in LinearKalman.
    # If propagator is set to propagate_LAI_narrowbandSAIL then the
    # prior in LinearKalman must be set to None because this propagator
    # includes a prior
    propagator = propagate_LAI_narrowbandSAIL

    parameter_list = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm',
                      'lai', 'ala', 'bsoil', 'psoil']

    #start_time = "2017001"
    #time_grid_start = datetime(2017, 1, 1)
    start_time = "2017100"
    time_grid_start = datetime(2017, 3, 3)
    #time_grid_start = datetime(2017, 5, 24)
    num_days = 250

    Log.info("propagator = {}".format(propagator))
    Log.info("Qfactor = {}".format(Qfactor))
    Log.info("start_time = {}".format(start_time))

    path = "/home/npounder/output/planet/kafkaout_{}".format(runname)
    if not os.path.exists(path):
        mkdir_p(path)

    Log.info("output path = {}".format(path))
    emulator_folder = "/data/gp_emulators/prosail/planetlabs/"

    data_folder = "/data/001_planet_sentinel_study/planet/utm11n_sur_ref/"
    #data_folder = "/data/001_planet_sentinel_study/planet/utm11n_sur_ref/subset2/"
    #data_folder = "/home/npounder/repositories/python3/KaFKA-InferenceEngine/subset/"

    #state_mask = "/data/001_planet_sentinel_study/planet/utm11n_sur_ref/field_sites.tif"
    #state_mask = "/home/npounder/repositories/python3/KaFKA-InferenceEngine/dataCleansing/Window200Pix.tif"
    state_mask = "/home/npounder/repositories/python3/KaFKA-InferenceEngine/dataCleansing/GoodFieldMask.tif"

    planet_observations = PlanetlabsObservations(data_folder,
                                                 emulator_folder,
                                                 state_mask)

    projection, geotransform = planet_observations.define_output()

    output = KafkaOutput(parameter_list, geotransform,
                         projection, path)

    # If using a separate prior then this is passed to LinearKalman
    # Otherwise this is just used to set the starting state vector.
    the_prior = SAILPrior(parameter_list, state_mask)

    g = gdal.Open(state_mask)
    mask = g.ReadAsArray().astype(np.bool)

    # prior = None when using propagate_LAI_narrowbandSAIL
    kf = LinearKalman(planet_observations, output, mask,
                      create_prosail_observation_operator,
                      parameter_list,
                      state_propagation=propagator,
                      prior=None,#the_prior,
                      linear=False)
    '''
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
    '''
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

