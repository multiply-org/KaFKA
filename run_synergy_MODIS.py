#!/usr/bin/env python

import logging
logging.basicConfig(
    level=logging.getLevelName(logging.DEBUG),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="logfiles/nebraska_2017_noprop_p1-5_e1.log")
import os
from datetime import datetime, timedelta
import errno

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

    runname = 'nebraska_2017_noprop_p1-5_e1'  #Used in output directory as a unique identifier
    #runname = 'modis_tstepbug'  #Used in output directory as a unique identifier
    Qfactor = 0.05

    # To run without propagation set propagator to None and set a
    # prior in LinearKalman.
    # If propagator is set to propagate_LAI_broadbandSAIL then the
    # prior in LinearKalman must be set to None because this propagator
    # includes a prior
    propagator = None #propagate_LAI_broadbandSAIL
    parameter_list = ["w_vis", "x_vis", "a_vis",
                      "w_nir", "x_nir", "a_nir", "TeLAI"]

    ## parameters for Harvard Forest data.
    #tile = "h12v04"
    #start_time = "2009001"
    #time_grid_start = datetime(2009, 1, 1)
    #time_grid_start = datetime(2010, 3, 22)
    #num_days = 366
    #mcd43a1_dir="/data/MODIS/h12v04/MCD43"
    #period=1 #This data exists every 8 days so period=1 gives 8 day data
    #masklim = ((1740, 1840), (1590, 1690))

    ## parameters for Nebraska data.
    tile = "h10v04"
    start_time = "2017001"
    time_grid_start = datetime(2017, 1, 1)
    ####time_grid_start = datetime(2008, 3, 5)
    num_days = 366
    mcd43a1_dir="/data/MODIS/h10v04/MCD43_passedMask_fluxnetchip"
    period=1 #This data exists every 8 days so period=1 gives 8 day data
    masklim = ((2100, 2150), (1750, 1800))
    ###masklim = ((880, 1065), (880, 1065))

    ## parameters for Bondville data.
    #tile = "h11v04"
    #start_time = "2006001"
    #time_grid_start = datetime(2006, 1, 1)
    #num_days = 366
    #mcd43a1_dir="/data/MODIS/h11v04/MCD43"
    ## Bondville chip
    #masklim = ((2200, 2400), (450, 700))   # Bondville, h11v04

    ## Parameters for Spanish tile
    #tile = "h17v05"      # Spain
    #start_time = "2017001"    # Spain
    #time_grid_start = datetime(2017, 1, 1)
    #num_days = 366
    #mcd43a1_dir="/data/MODIS/h17v05/MCD43"
    ## chips in h17v05 Spain, select one
    #masklim = ((680, 710), (1200, 1220))     # Tiny test
    #masklim = ((650, 730), (1180, 1280))     # Arros, rice
    #masklim = ((900,940), (1300,1340)) = True # Alcornocales
    #masklim = ((640,700), (1400,1500)) = True # Campinha



    path = "/home/npounder/output/kafka/validation/kafkaout_{}".format(runname)
    if not os.path.exists(path):
        mkdir_p(path)

    emulator = "./SAIL_emulator_both_500trainingsamples.pkl"


    mask = np.zeros((2400,2400),dtype=np.bool8)
    mask[masklim[0][0]:masklim[0][1],
         masklim[1][0]:masklim[1][1]] = True

    # The 'period' keyword selects every nth observation. So if your data
    # is every 8 days then period=1 will give an observation every 8 days.
    # If your data is daily then set period=8 to get 8 day data
    bhr_data = BHRObservations(emulator, tile, mcd43a1_dir, start_time,
                               end_time=None, mcd43a2_dir=None, period=period)

    Log.info("propagator = {}".format(propagator))
    Log.info("Qfactor = {}".format(Qfactor))
    Log.info("tile = {}".format(tile))
    Log.info("start_time = {}".format(start_time))
    Log.info("mask = {}".format(masklim))
    Log.info("output path = {}".format(path))

    projection, geotransform = bhr_data.define_output()

    output = KafkaOutput(parameter_list, geotransform, projection, path)

    # If using a separate prior then this is passed to LinearKalman
    # Otherwise this is just used to set the starting state vector.
    the_prior = JRCPrior(parameter_list, mask)

    # prior = None when using propagate_LAI_broadbandSAIL
    kf = LinearKalman(bhr_data, output, mask,
                      create_nonlinear_observation_operator,parameter_list,
                      state_propagation=propagator,
                      prior=the_prior,#None,
                      linear=False)

    # Get starting state... We can request the prior object for this
    x_forecast, P_forecast_inv = the_prior.process_prior(None)

    # Inflation amount for propagation
    Q = np.zeros_like(x_forecast)
    Q[6::7] = Qfactor

    kf.set_trajectory_model()
    kf.set_trajectory_uncertainty(Q)

    # This determines the time grid of the retrieved state parameters
    time_grid = []
    for x in range(0, num_days, 8):
        time_grid.append(time_grid_start + timedelta(days=x))

    kf.run(time_grid, x_forecast, None, P_forecast_inv, iter_obs_op=True)