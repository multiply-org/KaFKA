import datetime

from Planetlabs_Observations import PlanetlabsObservations


if __name__ == "__main__":

    start_time = "2017001"

    emulator_folder = "/data/gp_emulators/prosail/planetlabs/"

    data_folder = "/data/001_planet_sentinel_study/planet/utm11n_sur_ref/"

    state_mask = "/data/001_planet_sentinel_study/planet/utm11n_sur_ref/field_sites.tif"

    planet_observations = PlanetlabsObservations(data_folder,
                                                 emulator_folder,
                                                 state_mask)

    planet = planet_observations.get_band_data(datetime.datetime
                                               (2017, 4, 4, 18, 1, 19), 1)

    print(planet)
