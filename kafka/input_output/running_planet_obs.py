import Planetlabs_Observations


if __name__ == "__main__":

    start_time = "2017001"

    emulator_folder = "/home/glopez/Multiply/src/py36/emus/sail"

    data_folder = "/data/001_planet_sentinel_study/planet/utm11n_sur_ref/"

    state_mask = "/data/001_planet_sentinel_study/planet/utm11n_sur_ref/state_mask.tif"

    planet_observations = Planetlabs_Observations(data_folder,
                                                  emulator_folder,
                                                  state_mask)
