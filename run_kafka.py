import logging
import os
import datetime as dt
import numpy as np
import gdal

try:
    from multiply_prior_engine import PriorEngine
except ImportError:
    pass

from kafka.input_output import Sentinel2Observations, KafkaOutput
from kafka import LinearKalman
# from kafka.inference.narrowbandSAIL_tools import propagate_LAI_narrowbandSAIL
from kafka.inference import create_prosail_observation_operator
from kafka.inference.narrowbandSAIL_tools import SAILPrior
# from kafka.inference.temporal_prior import TemporalSAILPrior
# from kafka.inference.propagate_phenology import TrajectoryFromPrior

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == exc.errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def run_kafka(data_folder, state_mask, out_path, start_time, ndays):

    if not os.path.exists(out_path):
        mkdir_p(out_path)
    # Set up logging


    Log = logging.getLogger(__name__+".kafka_test_x.py")

    _, fname = os.path.split(state_mask)
    logfile = '{}.log'.format(fname[:-4])

    logging.basicConfig(level=logging.getLevelName(logging.DEBUG),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename="{}/{}".format(out_path, logfile))

    Qfactor = 0.1

    # To run without propagation set propagator to None and set a
    # prior in LinearKalman.
    # If propagator is set to propagate_LAI_narrowbandSAIL then the
    # prior in LinearKalman must be set to None because this propagator
    # includes a prior
    propagator = None #propagate_LAI_narrowbandSAIL

    parameter_list = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm',
                      'lai', 'ala', 'bsoil', 'psoil']

    Log.info("propagator = {}".format(propagator))
    #Log.info("Qfactor = {}".format(Qfactor))
    Log.info("start_time = {}".format(start_time))

    Log.info("output path = {}".format(out_path))

    emulator_folder = "/home/glopez/Multiply/src/py36/emus/sail"

    s2_observations = Sentinel2Observations(data_folder,
                                            emulator_folder,
                                            state_mask)
    s2_observations.check_mask()
    projection, geotransform = s2_observations.define_output()

    output = KafkaOutput(parameter_list, geotransform,
                         projection, out_path)

    the_prior = SAILPrior(parameter_list, state_mask)

    g = gdal.Open(state_mask)
    mask = g.ReadAsArray().astype(np.bool)

    # prior = None when using propagate_LAI_narrowbandSAIL
    kf = LinearKalman(s2_observations, output, mask,
                      create_prosail_observation_operator,
                      parameter_list,
                      state_propagation=propagator,
                      prior=the_prior,
                      linear=False)

    # This determines the time grid of the retrieved state parameters
    time_grid = list((start_time + dt.timedelta(days=x)
                     for x in range(0, ndays, 1)))
    # Get starting state... We can request the prior object for this
    x_forecast, P_forecast_inv = the_prior.process_prior(time_grid[0])

    # Inflation amount for propagation
    Q = np.zeros_like(x_forecast)
    Q[6::10] = Qfactor
    kf.set_trajectory_uncertainty(Q)

    kf.set_trajectory_model()
    kf.run(time_grid, x_forecast, None, P_forecast_inv,
           iter_obs_op=True)