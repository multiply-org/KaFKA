from collections import namedtuple

#highres_data = namedtuple('planetlabs_data',
#                          'observations uncertainty mask metadata emulator')


class HighresObservations(object):
    def __init__(self, observations):
        self.observations = observations
        self.dates = []
        self.obs_index = {}
        self.bands_per_observation = {}
        for i, obs in enumerate(observations):
            for date in obs.dates:
                self.dates.append(date)
                self.obs_index[date] = i
                self.bands_per_observation[date] = obs.bands_per_observation[date]

    def get_band_data(self, timestep, band):
        obs_index = self.obs_index[timestep]
        data = self.observations[obs_index].get_band_data(timestep, band)

        return data
