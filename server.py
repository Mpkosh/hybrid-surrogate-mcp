#!/usr/bin/env python3
from fastmcp import FastMCP
from hybrid_surr import plot_hyb,plot_funcs,aux_f

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
folder_main = 'hybrid_surr/'

@mcp.tool()
def run_hybrid_once(sigma: float=0.3, gamma: float=0.2, 
               perc_switch: float=0.01, stoch: int=10, 
               topology: Literal["ba", "sw"]='ba', 
                beta_pred: Literal['last value',
                        'expanding mean last value',
                        'median beta', 'regression beta', 
                        'lstm']='median beta',
              seed_numbers:list[int]=[0,10]) -> dict:
    if topology=='ba':
        suff_m='ba100k'
        #suff='ba100k_10'
        seed_dirs = '../long_sw_100000_fin/'
        sw = pd.read_csv(f'{folder_main}/aux_hyb/ba_test_files.csv').values
    if topology=='sw':
        suff_m='sw100k'
        #suff='sw100k_10'
        seed_dirs='../new_ba_100000/'
        sw = pd.read_csv(f'{folder_main}/aux_hyb/test_files.csv').values
        
    seed_numbers = sw[::10][seed_numbers]
    m_folder=f'{folder_main}/aux_hyb/'
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
               topology: Literal["ba", "sw"]='ba', 
              methods: list[Literal['last value',
                        'expanding mean last value',
                        'median beta', 'regression beta', 
                        'lstm']]=['median beta'],
              seed_numbers: list[int]=[0,10]) -> dict:
    if topology=='ba':
        suff_m='ba100k'
        suff='ba100k_10'
        seed_dirs = '../long_sw_100000_fin/'
        sw = pd.read_csv(f'{folder_main}/aux_hyb/ba_test_files.csv').values
    if topology=='sw':
        suff_m='sw100k'
        suff='sw100k_10'
        seed_dirs='../new_ba_100000/'
        sw = pd.read_csv(f'{folder_main}/aux_hyb/test_files.csv').values
    seed_numbers = sw[::10][seed_numbers]
    #methods = ['last value','expanding mean last value',
    #           'median beta','regression beta', 'lstm']
    plot_hyb.apply_methods(seed_dirs=seed_dirs,
              seed_numbers=seed_numbers, on_incidence=True,
              switch_on_incidence=False,
              methods=methods, show_fig_flag=True,
             is_filename=True, sigma=sigma, gamma=gamma, 
              perc_switch=perc_switch, stoch=stoch, m_folder=f'{folder_main}/aux_hyb/',
             suff_m=suff_m, suff=suff)
    
    #plt.savefig('ii.png', bbox_inches='tight')
    
    metadata = {
        ".": 1,
        #**artifact_metadata,
    }
    return {"answer": f'Ran the hybrid model and uploaded the figure {11} to {11}', 
            "metadata": metadata}


@mcp.tool()
def calibrate_model_complete_data(model_name:Literal["hybrid", "network",'surrogate']='hybrid',
                    n_hyb_runs:int=1, nth:int=5,
                                  true_tau:float = 0.1,
                              true_alpha:float = 0.95,
                                 topology:Literal["ba","sw"]='ba',) -> dict:

    gamma = 0.3 # latent period rate
    delta = 0.2 # recovery rate
    folder=f'{folder_main}/calibr/'
    
    observed_data = pd.read_csv(folder+f'observed_incidence_a{true_alpha}_b{true_tau}.csv'
                           ).iloc[:100]
    observed_data.columns=['incidence']

    # _________
    with_switch=np.array(True) 
    num_runs=[1]
    frac=[0.05]
    n_nodes=100000

    if model_name=='hybrid':
        model_str = 'hyb'
    elif model_name=='network':
        model_str = 'net'
    elif model_name=='surrogate':
        model_str = 'surr'
        
    top=topology
    koeff=1
    shift=0
    network_params=[with_switch,num_runs,frac,gamma,
                    delta,n_nodes,top,koeff,shift]

    model_idata=f"{top}_{model_str}_a{true_alpha}_b{true_tau}.nc"
    idata = az.from_netcdf(folder+model_idata)
    if model_name=='surrogate':
        idata = idata.rename({"beta": "tau"})
        
    beta_mode, alpha_mode = plot_funcs.plot_calib(observed_data, idata, 
               true_tau, true_alpha, network_params, 
               n_hyb_runs=n_hyb_runs, nth=nth)
    #plt.savefig(f'results/ba_hybrid.pdf', format='pdf', bbox_inches='tight')
    

    metadata = {
        "beta_mode":beta_mode,
        "alpha_mode":alpha_mode,
        #**artifact_metadata,
    }
    return {"answer": f'Calibrated the {model_name} model and uploaded the figure {11} to {11}', 
            "metadata": metadata}


@mcp.tool()
def calibrate_model_complete_data_3in1(n_hyb_runs:int=1, nth:int=5,
                                       true_tau:float = 0.1,
                              true_alpha:float = 0.95,
                                      topology:Literal["ba","sw"]='ba',) -> dict:
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    labels = ['('+alphabet[index] + ')' for index in range(len(alphabet))]
    
    fig = plt.figure(figsize=(20,10))
    # adding gridspec
    gs = fig.add_gridspec(2, 3, hspace=0.2, width_ratios=[1,1,1],
                         height_ratios=[1,1.25])
    
    # curves subplots
    gs00 = gs[0,0].subgridspec(1, 2, width_ratios=[4, 1],
                               wspace=0.)
    gs01 = gs[0,1].subgridspec(1, 2, width_ratios=[4, 1],
                              wspace=0.)
    gs02 = gs[0,2].subgridspec(1, 2, width_ratios=[4, 1],
                              wspace=0.)
    
    # ufo subplots
    gs10 = gs[1,0].subgridspec(2, 2, wspace=0, hspace=0.,
                            width_ratios=[4, 1],
                            height_ratios=[1, 4])
    gs11 = gs[1,1].subgridspec(2, 2, wspace=0, hspace=0.,
                            width_ratios=[4, 1],
                            height_ratios=[1, 4])
    gs12 = gs[1,2].subgridspec(2, 2, wspace=0, hspace=0.,
                            width_ratios=[4, 1],
                            height_ratios=[1, 4])
    

    gamma = 0.3 # latent period rate
    delta = 0.2 # recovery rate
    folder=f'{folder_main}/calibr/'
    
    observed_data = pd.read_csv(folder+f'observed_incidence_a{true_alpha}_b{true_tau}.csv'
                           ).iloc[:100]
    observed_data.columns=['incidence']

    # _________
    with_switch=np.array(True) 
    num_runs=[1]
    frac=[0.05]
    n_nodes=100000

    top=topology
    koeff=1
    shift=0
    network_params=[with_switch,num_runs,frac,gamma,
                    delta,n_nodes,top,koeff,shift]
    idatas = [az.from_netcdf(folder+\
                    f"{top}_{model_str}_a{true_alpha}_b{true_tau}.nc") \
                  for model_str in ['net','hyb','surr']
                 ]
    # surr model's idata was saved with other arguments
    idatas[-1] = idatas[-1].rename({"beta": "tau"})
    
    beta_modes, alpha_modes = [],[]
    for gs_0i, gs_1i, idata_i,idx  in zip([gs00,gs01,gs02],
                                 [gs10,gs11,gs12],
                                 idatas,
                                     np.arange(3)):
    
        if idx==1:
            frac=[0.05]
            with_switch=np.array(True) 
            network_params=[with_switch,num_runs,frac,gamma,
                            delta,n_nodes,top,koeff,shift]
        elif idx == 2:
            num_runs = [0]
            frac=[1]
            with_switch=np.array(False) 
            network_params=[with_switch,num_runs,frac,gamma,
                            delta,n_nodes,top,koeff,shift]
    
        ax_curves = fig.add_subplot(gs_0i[0])
        ax_up = fig.add_subplot(gs_1i[0])
        ax_scatter = fig.add_subplot(gs_1i[2], sharex=ax_up)
        ax_right = fig.add_subplot(gs_1i[3], sharey=ax_scatter)
    
        beta_mode, alpha_mode = plot_funcs.plot_calib(observed_data, idata_i, 
                       true_tau, true_alpha, 
                       network_params, pred=False,nth=nth,
                              n_hyb_runs=n_hyb_runs,
                             ax_curves=[ax_curves],
                             ax_kde=[ax_up,ax_scatter,ax_right])
        fontsize=14
        ax_curves.annotate(labels[idx], xy=(0, 0), xycoords='axes fraction',
                               xytext=(-30, -50), textcoords='offset points',
                               fontsize=1.5*fontsize, ha='right', va='baseline')
        beta_modes.append(beta_mode)
        alpha_modes.append(alpha_mode)

    metadata = {
        "beta_modes":beta_modes,
        "alpha_modes":alpha_modes,
        #**artifact_metadata,
    }
    return {"answer": f'Calibrated three models and uploaded the figure {11} to {11}', 
            "metadata": metadata}
    

@mcp.tool()
def calibrate_model_forecast(model_name:Literal["hybrid",
                             'surrogate']='hybrid',
                    n_hyb_runs:int=1, nth:int=5,
                    start_pred:Literal["14b","7b","7a"]='14b',
                             topology:Literal["ba","sw"]='ba',
                             true_tau:float = 0.1,
                              true_alpha:float = 0.95,) -> dict:

    gamma = 0.3 # latent period rate
    delta = 0.2 # recovery rate
    folder=f'{folder_main}/calibr/'
    observed_data = pd.read_csv(folder+f'observed_incidence_a{true_alpha}_b{true_tau}.csv'
                           ).iloc[:100]
    observed_data.columns=['incidence']
    
    with_switch=np.array(False) 
    num_runs=[1]
    frac=[1]
    n_nodes=100000
    top=topology
    koeff=1
    shift=0
    network_params=[with_switch,num_runs,frac,
                    gamma,delta,n_nodes,top,koeff,shift]
    
    if model_name=='hybrid':
        model_str = 'hyb'
    elif model_name=='surrogate':
        model_str = 'surr'
        
    model_idata=f"{top}_{model_str}_a{true_alpha}_b{true_tau}_{start_pred}.nc"
    idata = az.from_netcdf(folder+model_idata)

    beta_mode, alpha_mode = plot_funcs.plot_calib(observed_data, idata, 
                       true_tau, true_alpha, 
                       network_params, pred=True, nth = nth)
    
    #plt.savefig(f'results/ba_pred_eps1000.pdf', format='pdf', bbox_inches='tight')
    metadata = {
        "beta_mode":beta_mode,
        "alpha_mode":alpha_mode,
        #**artifact_metadata,
    }
    return {"answer": f'Forecasted with the {model_name} model and uploaded the figure {11} to {11}', 
            "metadata": metadata}


@mcp.tool()
def calibrate_model_forecast_3in1(model_name:Literal["hybrid",
                             'surrogate']='hybrid',
                    n_hyb_runs:int=1,nth:int=5,
                                  true_tau:float = 0.1,
                              true_alpha:float = 0.95,
                                 topology:Literal["ba","sw"]='ba',) -> dict:
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    labels = ['('+alphabet[index] + ')' for index in range(len(alphabet))]

    gamma = 0.3 # latent period rate
    delta = 0.2 # recovery rate
    folder=f'{folder_main}/calibr/'
    observed_data = pd.read_csv(folder+f'observed_incidence_a{true_alpha}_b{true_tau}.csv'
                           ).iloc[:100]
    observed_data.columns=['incidence']
    
    with_switch=np.array(False) 
    num_runs=[1]
    frac=[1]
    n_nodes=100000
    top=topology
    koeff=1
    shift=0
    network_params=[with_switch,num_runs,frac,
                    gamma,delta,n_nodes,top,koeff,shift]
    
    if model_name=='hybrid':
        model_str = 'hyb'
    elif model_name=='surrogate':
        model_str = 'surr'

    fig = plt.figure(figsize=(20,10))
    # adding gridspec
    gs = fig.add_gridspec(2, 3, hspace=0.2, width_ratios=[1,1,1],
                         height_ratios=[1,1.25])
    
    # curves subplots
    gs00 = gs[0,0].subgridspec(1, 2, width_ratios=[4, 1],
                               wspace=0.)
    gs01 = gs[0,1].subgridspec(1, 2, width_ratios=[4, 1],
                              wspace=0.)
    gs02 = gs[0,2].subgridspec(1, 2, width_ratios=[4, 1],
                              wspace=0.)
    
    # ufo subplots
    gs10 = gs[1,0].subgridspec(2, 2, wspace=0, hspace=0.,
                            width_ratios=[4, 1],
                            height_ratios=[1, 4])
    gs11 = gs[1,1].subgridspec(2, 2, wspace=0, hspace=0.,
                            width_ratios=[4, 1],
                            height_ratios=[1, 4])
    gs12 = gs[1,2].subgridspec(2, 2, wspace=0, hspace=0.,
                            width_ratios=[4, 1],
                            height_ratios=[1, 4])

    idatas = [az.from_netcdf(folder+\
                        f"{top}_{model_str}_a{true_alpha}_b{true_tau}_{start_pred}.nc") \
              for start_pred in ["14b","7b","7a"]
             ]
    beta_modes, alpha_modes = [],[]
    for gs_0i, gs_1i, idata_i,idx  in zip([gs00,gs01,gs02],
                                 [gs10,gs11,gs12],
                                 idatas,
                                     np.arange(3)):
        
        ax_curves = fig.add_subplot(gs_0i[0])
    
        ax_up = fig.add_subplot(gs_1i[0])
        ax_scatter = fig.add_subplot(gs_1i[2], sharex=ax_up)
        ax_right = fig.add_subplot(gs_1i[3], sharey=ax_scatter)
    
        beta_mode, alpha_mode = plot_funcs.plot_calib(observed_data, idata_i, 
                       true_tau, true_alpha, 
                       network_params, pred=True,
                              nth=nth,
                             ax_curves=[ax_curves],
                             ax_kde=[ax_up,ax_scatter,ax_right])
        beta_modes.append(beta_mode)
        alpha_modes.append(alpha_mode)
        fontsize=14
        ax_curves.annotate(labels[idx], xy=(0, 0), xycoords='axes fraction',
                               xytext=(-30, -50), textcoords='offset points',
                               fontsize=1.5*fontsize, ha='right', va='baseline')
        
    metadata = {
        "beta_modes":beta_modes,
        "alpha_modes":alpha_modes,
        #**artifact_metadata,
    }
    return {"answer": f'Forecasted with the {model_name} model and uploaded the figure {11} to {11}', 
            "metadata": metadata}
    

@mcp.tool()
def plot_forecast_peak_errors(start_preds:list[Literal["14b","7b","7a"]]=\
                              ["14b","7b","7a"],
                              true_tau:float = 0.1,
                              true_alpha:float = 0.95,
                              topology:Literal["ba","sw"]='ba',
                             ) -> dict:
    top=topology
    folder = f'{folder_main}/calibr/'
    observed_data = pd.read_csv(folder+f'observed_incidence_a{true_alpha}_b{true_tau}.csv'
                               ).iloc[:100]
    observed_data.columns=['incidence']
    model_strs = ['hyb','surr']
    idatas = [[az.from_netcdf(folder+\
                    f"{top}_{model_str}_a{true_alpha}_b{true_tau}_{start_pred}.nc") \
                  for model_str in model_strs
                ] for start_pred in start_preds
             ]
    
    aux_f.create_peak_plot(folder_name=folder,
                     observed_data = observed_data,
                       idatas = idatas,
                     with_outliers=True, same_lims=True, 
                     figsize=(3.6*len(idatas),4), x_lim = (-30, 30), y_lim = (0, 1.5),
                     alpha_m=0.08, alpha_area=0.3, save=False)
    metadata = {
        #**artifact_metadata,
    }
    return {"answer": f'Uploaded the figure {11} to {11}', 
            "metadata": metadata}


@mcp.tool()
def plot_heatmap_r2(topology:Literal["ba","sw"]='ba',
                    ) -> dict:
    switch_perc = 5
    metric = 'r2'
    switch='fraq_people'
    fin_inc = aux_f.df_metrics(folder_name=f'{folder_main}/aux_hyb/',
                        top_name=f'new_{topology}_100000', 
                       test_suff=f'{topology}_', 
                       switch=switch,
                       with_inc=True, 
                       trim=False, suff=f'_{topology}100k_sI_fullR_{switch_perc}')
    
    aux_f.metric_hmaps(folder_name=f'{folder_main}/aux_hyb/',
                       fin=fin_inc, met=f'{metric}_Inc', 
                       suff=f'_{topology}_sI_{switch_perc}', 
                       exclude=['Cumulative Average',
                                'Exponential Decay'],
                        save=False)
    metadata = {
        #**artifact_metadata,
    }
    return {"answer": f'Uploaded the figure {11} to {11}', 
            "metadata": metadata}


@mcp.tool()
def plot_heatmap_switch(topology:Literal["ba","sw"]='ba',
                    ) -> dict:
    switch_perc = 5
    switch='fraq_people'
    fin_inc = aux_f.df_metrics(folder_name=f'{folder_main}/aux_hyb/',
                        top_name=f'new_{topology}_100000', 
                       test_suff=f'{topology}_', 
                       switch=switch,
                       with_inc=True, 
                       trim=False, suff=f'_{topology}100k_sI_fullR_{switch_perc}')
    
    aux_f.smth_hmaps(fin_inc, 21)
    
    metadata = {
        #**artifact_metadata,
    }
    return {"answer": f'Uploaded the figure {11} to {11}', 
            "metadata": metadata}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7331, path="/mcp")