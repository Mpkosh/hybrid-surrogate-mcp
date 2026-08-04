#!/usr/bin/env python3
from fastmcp import FastMCP
from hybrid_surr import plot_hyb,plot_funcs

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import logging
from typing import Any, Literal
#from io import BytesIO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Hybrid Surrogate Server")


@mcp.tool()
def use_hybrid_model(task: str, s3_keys: list[str]) -> dict:
    return {'answer': ''}


@mcp.tool()
def use_surrogate_model(s3_keys: list[str]) -> dict:
    """
    ...
    
    Args:
        s3_keys (list[str]): The list of S3 keys for the uploaded papers.
        config (RunnableConfig): The configuration for the runnable.


    Returns:
        dict: A dictionary containing the answer ... and metadata 
              about the request (e.g., model used, token count).  
              Returns an error message if ...
    """
    logger.info('Running use_surrogate_model tool...')
    try:
        a=3
        return a
    except Exception as e:
        logger.error(f'use_surrogate_model ERROR: {e}')
        return {'answer': 'Could not '}
        
    metadata = {
        ".": 1,
        #**artifact_metadata,
    }
    return {"answer": f'...', 
            "metadata": metadata}

# _______ plots:
@mcp.tool()
def orig_data_plots(s3_keys: list[str]) -> dict:
    return {'answer': ''}


@mcp.tool()
def run_hybrid_once(sigma: float=0.3, gamma: float=0.2, 
               perc_switch: float=0.01, stoch: int=10, 
               net_type: Literal["ba", "sw"]='ba', 
                beta_pred: Literal['last value',
                        'expanding mean last value',
                        'median beta', 'regression beta', 
                        'lstm']='median beta',
              seed_numbers:list[int]=[0,10]) -> dict:
    if net_type=='ba':
        suff_m='ba100k'
        #suff='ba100k_10'
        seed_dirs = '../long_sw_100000_fin/'
        sw = pd.read_csv('hybrid_surr/aux_hyb/ba_test_files.csv').values
    if net_type=='sw':
        suff_m='sw100k'
        #suff='sw100k_10'
        seed_dirs='../new_ba_100000/'
        sw = pd.read_csv('hybrid_surr/aux_hyb/test_files.csv').values
        
    seed_numbers = sw[::10][seed_numbers]
    m_folder='hybrid_surr/aux_hyb/'
    if 'median' in beta_pred:
        model_path = f'{m_folder}{suff_m}_median_beta.csv'
    elif 'regression beta' in beta_pred:
        model_path = f'{m_folder}{suff_m}_regression_bt.joblib'
    elif 'lstm' in beta_pred:
        model_path = f'{m_folder}{suff_m}_lstm_4_001_s10'   
    else:
        model_path=''
    show_fig_flag = True
    #methods = ['last value','expanding mean last value',
    #           'median beta','regression beta', 'lstm']
    all_rmse_I, all_rmse_Inc, all_rmse_Beta, \
    all_r2, all_r2_Inc, all_r2_full, all_r2_Inc_full,\
    all_peak, \
        execution_time, start_days = plot_hyb.main_f(I_prediction_method='seir', 
                            count_stoch_line=stoch, 
                            beta_prediction_method=beta_pred, 
                            type_start_day='fraq_people', 
                            seed_numbers=seed_numbers, 
                            show_fig_flag=show_fig_flag,
                            seed_dirs=seed_dirs, 
                            sigma=sigma, gamma=gamma, 
                            ax=None, model_path=model_path,
                            perc_switch=perc_switch,
                            is_filename=True,
                            on_incidence=True,
                            switch_on_incidence=False,
                            topology=net_type)
    
    all_peak = pd.DataFrame(all_peak, 
                    columns=['actual_peak_I', 'predicted_peak_I', 
                            'actual_peak_Inc', 'predicted_peak_inc',
                            'actual_peak_day', 'predicted_peak_day',
                            'actual_peak_day_Inc', 'predicted_peak_day_inc'])
    # creating a dataframe for peaks RMSE, predicted time, start day
    rmse_df = pd.DataFrame({
        'rmse_I': all_rmse_I,
        'rmse_Inc': all_rmse_Inc,
        'rmse_Beta': all_rmse_Beta,
        'r2': all_r2,
        'r2_Inc': all_r2_Inc,
        'r2_full': all_r2_full,
        'r2_Inc_full': all_r2_Inc_full,
        'time_predict': execution_time,
        'fraq_people': start_days})

    # merging dataframes
    results = pd.concat([rmse_df, all_peak], axis=1)
    
    metadata = {
        ".": 1,
        #**artifact_metadata,
    }
    return {"answer": f'Ran the hybrid model and uploaded the figure {11} to {11}', 
            "metadata": metadata}

@mcp.tool()
def run_hybrid(sigma: float=0.3, gamma: float=0.2, 
               perc_switch: float=0.01, stoch: int=100, 
               net_type: Literal["ba", "sw"]='ba', 
              methods: list[Literal['last value',
                        'expanding mean last value',
                        'median beta', 'regression beta', 
                        'lstm']]=['median beta'],
              seed_numbers: list[int]=[0,10]) -> dict:
    if net_type=='ba':
        suff_m='ba100k'
        suff='ba100k_10'
        seed_dirs = '../long_sw_100000_fin/'
        sw = pd.read_csv('hybrid_surr/aux_hyb/ba_test_files.csv').values
    if net_type=='sw':
        suff_m='sw100k'
        suff='sw100k_10'
        seed_dirs='../new_ba_100000/'
        sw = pd.read_csv('hybrid_surr/aux_hyb/test_files.csv').values
    seed_numbers = sw[::10][seed_numbers]
    #methods = ['last value','expanding mean last value',
    #           'median beta','regression beta', 'lstm']
    plot_hyb.apply_methods(seed_dirs=seed_dirs,
              seed_numbers=seed_numbers, on_incidence=True,
              switch_on_incidence=False,
              methods=methods, show_fig_flag=True,
             is_filename=True, sigma=sigma, gamma=gamma, 
              perc_switch=perc_switch, stoch=stoch, m_folder='hybrid_surr/aux_hyb/',
             suff_m=suff_m, suff=suff)
    
    #plt.savefig('ii.png', bbox_inches='tight')
    
    metadata = {
        ".": 1,
        #**artifact_metadata,
    }
    return {"answer": f'Ran the hybrid model and uploaded the figure {11} to {11}', 
            "metadata": metadata}
    
@mcp.tool()
def run_surrogate(s3_keys: list[str]) -> dict:
    metadata = {
        ".": 1,
        #**artifact_metadata,
    }
    return {"answer": f'Ran the hybrid model and uploaded the figure {11} to {11}', 
            "metadata": metadata}

@mcp.tool()
def plot_calibrated_hybrid(s3_keys: list[str]) -> dict:
    return {'answer': ''}
    
@mcp.tool()
def calibrate_model_complete_data(model_name:Literal["hybrid", "network",'surrogate']='hybrid',
                    n_hyb_runs:int=1) -> dict:
    true_tau = 0.1
    true_alpha = 0.95
    init_inf = 1e-4
    gamma = 0.3 # latent period rate
    delta = 0.2 # recovery rate
    folder='hybrid_surr/calibr/'
    
    observed_data = pd.read_csv(folder+'incidence_synthetic_curve.csv'
                           ).iloc[:100]
    observed_data.columns=['incidence']

    # _________
    with_switch=np.array(True) 
    num_runs=[1]
    frac=[0.05]
    gamma = 0.3
    delta=0.2
    n_nodes=100000
    
    draws = 600
    chains = 3

    if model_name=='hybrid':
        model_idata=f"ba_hyb_{draws}_{chains}_s3_r3.nc"
    elif model_name=='network':
        model_idata=f"ba_net_{draws}_{chains}.nc"
    else:
        model_idata=f"ba_surr.nc"
        
    top=''
    koeff=1
    shift=0
    network_params=[with_switch,num_runs,frac,gamma,
                    delta,n_nodes,top,koeff,shift]
    idata = az.from_netcdf(folder+model_idata)
    if model_name=='surrogate':
        idata = idata.rename({"beta": "tau"})
        
    beta_mode, alpha_mode = plot_funcs.plot_calib(observed_data, idata, 
               true_tau, true_alpha, 
               network_params, n_hyb_runs=n_hyb_runs)
    #plt.savefig(f'results/ba_hybrid.pdf', format='pdf', bbox_inches='tight')
    

    metadata = {
        "beta_mode":beta_mode,
        "alpha_mode":alpha_mode,
        #**artifact_metadata,
    }
    return {"answer": f'Calibrated the {model_name} model and uploaded the figure {11} to {11}', 
            "metadata": metadata}

@mcp.tool()
def calibrate_model_forecast(model_name:Literal["hybrid", "network",'surrogate']='hybrid',
                    n_hyb_runs:int=1) -> dict:
    metadata = {
        ".": 1,
        #**artifact_metadata,
    }
    return {"answer": f'Calibrated the {model_name} model and uploaded the figure {11} to {11}', 
            "metadata": metadata}




if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7331, path="/mcp")