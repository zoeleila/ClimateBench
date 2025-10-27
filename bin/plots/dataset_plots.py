import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from funcs.settings import RAW_DIR, GRAPHS_DIR

def get_data(exp_path, var):
    data = xr.open_dataset(exp_path)[var]
    dim_to_reduce = [dim for dim in data.dims if dim not in ['time']]
    if dim_to_reduce:
        data = data.mean(dim=dim_to_reduce)
    data = data.values
    print(f"Data shape for {var} in {exp_path}: {data.shape}")
    return data

if __name__ == "__main__":
    # Example usage
    exps = ['1pctCO2', 
           'abrupt-4xCO2', 
           'historical', 
           'hist-aer',
           'hist-GHG',
           'ssp126',
           'ssp245',
           'ssp370',
           'ssp370-lowNTCF',
           'ssp585']
    
    dict = {'inputs' : {'CO2': {'name': 'Anthropogenic CO2', 
                                'unit': 'GtCO2 / year',
                                'range': (0,10000)},
                        'CH4': {'name': 'Anthropogenic CH4', 
                                'unit': 'GtCH4 / year',
                                'range': (0, 0.8)},
                        'SO2': {'name': 'Anthropogenic SO2', 
                                'unit': 'TgSO2 / year',
                                'range': (0, 8e-12)},
                        'BC': {'name': 'Anthropogenic Black Carbon', 
                                'unit': 'TgBC / year',
                                'range': (0, 5e-13)}},
            'outputs' : {'diurnal_temperature_range': {'name': 'Diurnal Temperature Range',
                                                    'unit': 'K',
                                                    'range': (-0.75, 0.2)},
                        'tas' : {'name': 'Surface Air Temperature',
                                    'unit': 'K',
                                    'range': (-1, 9)},
                        'pr': {'name': 'Precipitation',
                                'unit': 'mm/day',
                                'range': (-1e-6, 5e-6)},
                        'pr90': {'name': '90th Percentile of Precipitation',
                                'unit': 'mm/day',
                                'range': (-0.3e-5, 1.3e-5)}}}

    for name, vars in dict.items():
        fig, ax = plt.subplots(3, 4, figsize=(12, 7))
        for i, var in enumerate(vars.keys()):
            for exp in exps:
                exp_path = RAW_DIR / f'train_val/{name}_{exp}.nc'
                if exp == 'ssp245':
                    exp_path = RAW_DIR / f'test/{name}_{exp}.nc'
                data = get_data(exp_path, var)
                if exp in ['historical', 'hist-aer', 'hist-GHG']:
                    dates = np.arange(1850, 2015)
                    ax[0, i%4].plot(dates, data, label=exp)
                    ax[0, i%4].set_title(f"{vars[var]['name']} \n {vars[var]['unit']}", fontsize=14)
                    ax[0, 0].legend()
                    ax[0, i%4].set_ylim(vars[var]['range'])
                
                elif exp in ['ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF', 'ssp585']:
                    dates = np.arange(2015, 2101)
                    if data.shape[0] != len(dates):
                        dates = np.arange(2015, 2101)[:data.shape[0]]
                    if exp == 'ssp370-lowNTCF':
                        exp = 'ssp370-\nlowNTCF'
                    ax[1, i%4].plot(dates, data, label=exp)   
                    ax[1, 0].legend(loc='lower left')
                    ax[1, i%4].set_ylim(vars[var]['range'])

                elif exp in ['1pctCO2', 'abrupt-4xCO2']:
                    years = np.arange(len(data))
                    ax[2, i%4].plot(years, data, label=exp)
                    ax[2, 0].legend(loc ='lower left')
                    ax[2, i%4].set_ylim(vars[var]['range'])
                    
                
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / f'{name}.png')
            