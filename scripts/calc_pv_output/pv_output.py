from lib2to3.pgen2 import driver
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import pvlib
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

def get_weather(lat, lng):
    print('Getting weather...')
    start = time.time()

    weather = pvlib.iotools.get_pvgis_tmy(lat, lng,
                                        map_variables=True)[0]
    weather.index.name = "utc_time"
    
    end = time.time()
    print(f'Completed getting weather in {end-start}s')
    return weather

def calculate_PV(db, weather):
    coordinates = list(zip(
        db['lat'], db['lng'], db['height_mean'], db['slope_mean'], db['aspect_mean'], db['shading_mean']
        ))
    timezone = 'Etc/GMT+1'

    energies = []
    for location in coordinates:
        latitude, longitude, height, slope, aspect, shading = location
        
        location = Location(
            latitude,
            longitude,
            name='',
            altitude=height,
            tz=timezone,
        )
        mount = FixedMount(surface_tilt=slope, surface_azimuth=aspect)
        array = Array(
            mount=mount,
            module_parameters=module,
            temperature_model_parameters=temperature_model_parameters,
        )
        system = PVSystem(arrays=[array], inverter_parameters=inverter)
        mc = ModelChain(system, location)
        mc.run_model(weather)
        annual_energy = mc.results.ac.sum()
        energies.append(annual_energy*shading)
    
    return energies

def main():
    FOLDER_DIR = 'output/'
    OUTPUT_DIR = 'pv_output/'
    filesInFolder = glob(FOLDER_DIR + '*.gml')

    for path in filesInFolder:
        filename = Path(filesInFolder).stem
        print(f'Computing pv output for {filename}...')
        db = gpd.read_file(path, driver='GML')
        db = db.to_crs(4326)
        db['lng'] = db.geometry.centroid.x
        db['lat'] = db.geometry.centroid.y

        if 'slope_mean' not in db:
            db['slope_mean'] = [0] * len(db)
        if 'aspect_mean' not in db:
            db['aspect_mean'] = [180] * len(db)

        if len(db) > 500:
            joined_db = gpd.GeoDataFrame(columns = db.columns, geometry='geometry')
            db = np.array_split(db, 50)
            for sub_db in db:
                sub_db['pv_output'] = calculate_PV(sub_db)
                joined_db = joined_db.concat(pd.concat([joined_db, sub_db], ignore_index=True))
                joined_db.to_file(OUTPUT_DIR + filename + '.gml', driver='GML')
        else:
            db['pv_output'] = calculate_PV(db)
            db.to_file(OUTPUT_DIR + filename + '.gml', driver='GML')
        
        print(f'Completed computing pv output.')

if __name__ == "__main__":
    main()