#!/usr/bin/env python

import logging
logging.basicConfig(
    level=logging.getLevelName(logging.DEBUG),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="the_log.log")
import os
from datetime import datetime, timedelta

import numpy as np
import gdal
import osr

# from multiply.inference-engine blah blah blah
try:
    from multiply_prior_engine import PriorEngine
except ImportError:
    pass

from kafka.input_output import BHRObservations, KafkaOutput
from kafka import LinearKalman
from kafka.inference.broadbandSAIL_tools import propagate_LAI_broadbandSAIL
from kafka.inference import create_nonlinear_observation_operator
from kafka.inference.broadbandSAIL_tools import JRCPrior


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


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == "__main__":

    # Set up logging
    Log = logging.getLogger(__name__+".kafka_test_x.py")

    runname = 'Arros_0-25'  #Used in output directory as a unique identifier

    # To run without propagation set propagator to None and set a
    # prior in LinearKalman.
    # If propagator is set to propagate_LAI_broadbandSAIL then the
    # prior in LinearKalman must be set to None because this propagator
    # includes a prior
    propagator = propagate_LAI_broadbandSAIL
    parameter_list = ["w_vis", "x_vis", "a_vis",
                      "w_nir", "x_nir", "a_nir", "TeLAI"]

    ## parameters for Bondville data.
    #tile = "h11v04"      # Bondville
    #start_time = "2006001"    # Bondville
    #cd43a1_dir="/data/MODIS/h11v04/MCD43"
    ## Bondville chip
    #masklim = ((2200, 2450), (450, 700))   # Bondville, h11v04

    # Parameters for Spanish tile
    tile = "h17v05"      # Spain
    start_time = "2017001"    # Spain
    mcd43a1_dir="/data/MODIS/h17v05/MCD43"
    # chips in h17v05 Spain, select one
    masklim = ((650, 730), (1180, 1280))     # Arros, rice
    #masklim = ((900,940), (1300,1340)) = True # Alcornocales
    #masklim = ((640,700), (1400,1500)) = True # Campinha



    path = "/tmp/kafkaout_{}".format(runname)
    if not os.path.exists(path):
        mkdir_p(path)

    emulator = "./SAIL_emulator_both_500trainingsamples.pkl"


    mask = np.zeros((2400,2400),dtype=np.bool8)
    mask[masklim[0][0]:masklim[0][1],
         masklim[1][0]:masklim[1][1]] = True

    bhr_data = BHRObservations(emulator, tile, mcd43a1_dir, start_time,
                               end_time=None, mcd43a2_dir=None, period=8)

    Log.info("propagator = {}".format(propagator))
    Log.info("tile = {}".format(tile))
    Log.info("start_time = {}".format(start_time))
    Log.info("mask = {}".format(masklim))

    projection, geotransform = bhr_data.define_output()

    output = KafkaOutput(parameter_list, geotransform, projection, path)

    # If using a separate prior then this is passed to LinearKalman
    # Otherwise this is just used to set the starting state vector.
    the_prior = JRCPrior(parameter_list, mask)

    # prior = None when using propagate_LAI_broadbandSAIL
    kf = LinearKalman(bhr_data, output, mask,
                      create_nonlinear_observation_operator,parameter_list,
                      state_propagation=propagator,
                      prior=None,
                      linear=False)

    # Get starting state... We can request the prior object for this
    x_forecast, P_forecast_inv = the_prior.process_prior(None)

    Q = np.zeros_like(x_forecast)
    Q[6::7] = 0.25

    kf.set_trajectory_model()
    kf.set_trajectory_uncertainty(Q)

    # This determines the time grid of the retrieved state parameters
    base = datetime(2017, 1, 1)
    num_days = 366
    time_grid = []
    for x in range( 0, num_days, 8):
        time_grid.append(base + timedelta(days=x))

    kf.run(time_grid, x_forecast, None, P_forecast_inv, iter_obs_op=True)