import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import time 
import os

import pvlib
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain

# Set solar panel assumptions
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

def get_weather(lat, lng):
    """
    Dataframe of hourly weather at coordinates for several years.

    Inputs
    lat(float): Latitude of location
    lng(float): Longitude of location

    Output
    (Dataframe): Hourly weather data at coordinate location.
    
    """
    print('Getting weather...')
    start = time.time()

    weather = pvlib.iotools.get_pvgis_tmy(lat, lng,
                                        map_variables=True)[0]
    weather.index.name = "utc_time"
    
    end = time.time()
    print(f'Completed getting weather in {end-start}s')
    return weather

def calculate_PV(location, weather, slope, aspect):
    """
    Estimate solar panel output based on solar pv module, temperature modelling and weather.
    
    Inputs
    location(pvlib Location): Object with coordinates, altitude and timezone
    weather(Dataframe): Hourly weather values for several years
    slope(float): Angle in degrees the solar panel is tilted from horizontal
    aspect(float): Azimuth (orientation) of solar panel to the sun

    Output
    (float): Total solar panel output for a year in Whr/yr/m^2
    
    """
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

    return annual_energy

def avg_pv_output(location, weather):
    """
    Average solar panel output over different potential aspect and slope configurations. 

    Input
    location(pvlib Location): Object with coordinates, altitude and timezone
    weather(Dataframe): Hourly weather values for several years

    Output:
    (float): Average solar pv output over different possible solar panel setups in Whr/yr/m^2
    
    """
    print('Calculating solar output...')
    start = time.time()

    slope_vals = np.arange(0, 61, 1)
    aspect_vals = np.arange(0, 175, 5)

    total_energy, counter = 0, 0
    for slope in slope_vals:
        for aspect in aspect_vals:
            total_energy += calculate_PV(location, weather, slope, aspect)
            counter += 1
    avg_energy = total_energy/counter

    end = time.time()
    print(f"Completed calculating PV output of {avg_energy} in {end-start}s")

    return avg_energy

def roof_segment_output(db, weather, timezone='Etc/GMT+1'):
    """
    Estimated yearly solar panel output for each roof segment given its slope and aspect.

    Input
    db(DataFrame): Roof segment features
    weather(Dataframe): Hourly weather values for several years
    timezone(str)

    Output
    energies(list): Solar pv output estimates for roof segments
    
    """
    print('Calculating solar output...')
    start = time.time()

    energies = []
    for index, row in db.iterrows():
        location = Location(
            row['lat'],
            row['lng'],
            name='',
            altitude=row['height_mean'],
            tz=timezone
        )
        annual_energy = calculate_PV(location, weather, row['slope_mean'], row['aspect_mean'])
        energies.append(annual_energy)

    end = time.time()
    print(f"Completed calculating PV output of {annual_energy} in {end-start}s")

    return energies

def pv_no_DSM(path):
    """
    Estimated solar pv output for houses without a DSM.

    Input
    path(str): Path to building footprint vector layer

    Output
    db(DataFrame): Estimated average yearly pv output for each UPRN.
    
    """
    filename = Path(path).stem
    print(f'Computing pv output for {filename}...')
    db = gpd.read_file(path, driver='GeoJSON')
    db = db.to_crs(4326)
    db['lng'] = db.geometry.centroid.x
    db['lat'] = db.geometry.centroid.y

    latitude = db['lat'].mean()
    longitude = db['lng'].mean()
    avg_alt = 103 # average altitude of West Midlands

    location = Location(
                        latitude,
                        longitude,
                        name='',
                        altitude=avg_alt,
                        tz='Etc/GMT+1',
                    )
    weather = get_weather(latitude, longitude)

    pv_output = avg_pv_output(location, weather)
    db['pv_output'] = db['shading_mean'] * pv_output * db['calculatedAreaValue'] * 0.9

    # Select columns to keep 
    keep_cols = ['uprn', 'lng', 'lat', 'shading_mean', 'calculatedAreaValue', 'pv_output']
    db = pd.DataFrame(db)[keep_cols]

    OUTPUT_DIR = os.getcwd() + '\\output\\no_DSM\\'
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    db.to_csv(f"{OUTPUT_DIR}{filename}.csv", index=False)
    print(f'Completed computing pv output.')
    
    return db

def roof_segment(path):
    """
    Estimate pv output for roof segments using pvlib.

    Input
    path(str): Path to roof segment vector layer.

    Output
    db(DataFrame): Estimated yearly solar pv output for each UPRN.

    """
    filename = Path(path).stem
    print(f'Computing pv output for {filename}...')

    db = gpd.read_file(path, driver="GeoJSON")
    db = db.dropna()
    db = db.to_crs(4326)
    db['lng'] = db.geometry.centroid.x
    db['lat'] = db.geometry.centroid.y

    latitude = db['lat'].mean()
    longitude = db['lng'].mean()
    weather = get_weather(latitude, longitude)

    output = roof_segment_output(db, weather)
    db['pv_output'] = db['shading_mean'] * output * db['AREA'].apply(np.floor)
    db = db.groupby('uprn').agg(
                        {
                        'lat':np.average,
                        'lng':np.average,
                        'slope_mean':np.average,
                        'aspect_mean':np.average,
                        'shading_mean':np.average, 
                        'pv_output': 'sum', 
                        'AREA':'sum'
                        })

    OUTPUT_DIR = os.getcwd() + '\\output\\roof_segment\\'
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    db.to_csv(f"{OUTPUT_DIR}{filename}.csv", index=False)

    return db

def main():
    FOLDER_DIR = os.getcwd() + '\\scripts\\calc_shadow\\output\\'
    filesInFolder = glob(FOLDER_DIR + '*.geojson')
    for path in filesInFolder:
        roof_segment(path)

    FOLDER_DIR = os.getcwd() + '\\data\\output\\proxy_data\\'
    filesInFolder = glob(FOLDER_DIR + '*.geojson')
    for path in filesInFolder:
        pv_no_DSM(path)

if __name__ == "__main__":
    main()