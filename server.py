#!/usr/bin/env python3
from fastmcp import FastMCP
from hybrid_surr import plot_hyb,plot_funcs,aux_f,surr_funcs

import torch
from sklearn.metrics import r2_score#, root_mean_squared_error, top_k_accuracy_score
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Any, Literal
#from io import BytesIO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Hybrid Surrogate Server")
folder_main = 'hybrid_surr/'
folder_imgs = 'imgs/'
search_full=False


# --------- hybrid ----------
@mcp.tool()
def run_hybrid_once(sigma: float=0.3, gamma: float=0.2, 
               perc_switch: float=0.01, stoch: int=10, 
               topology: Literal["ba", "sw"]='ba', 
                beta_pred: Literal['last value',
                        'expanding mean last value',
                        'median beta', 'regression beta', 
                        'lstm']='median beta',
              seed_numbers:list[int]=[93,2390]) -> dict:
    """
    Runs the hybrid model: start with the network model, 
        switch to the hybrid model with a given method of beta (infection transmission rate) prediction. 
    It returns the figure of predicted time series for incidence and beta.

    Args:
        sigma (float): The duration of the latent period (where 1/sigma is period in days)
        gamma (float): The duration of the infectious period (where 1/gamma is period in days)
        perc_switch (float): Switch to the hybrid model when the fraction 
            of Infected reaches <perc_switch>.
        stoch (int): Number of hybrid model runs to calculate interval bounds.
        topology (str): Short name of the network topology. 
            'ba' - Barabasi-Albert, 'sw' - small world.
        beta_pred (str): Method for beta prediction.
            'last value' - take last known beta value,
            'expanding mean last value' - take the ,
            'median beta' - median value of beta on train samples, 
            'regression beta' - regression model trained for beta prediction, 
            'lstm' - long short term memory model trained for beta prediction.
        seed_numbers (list): Indexes of test samples.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure, 
            coefficient of determination for test samples, the days of switch for test samples).
            
    """
    
    if topology=='ba':
        suff_m='ba100k'
        #suff='ba100k_10'
        if search_full:
            seed_dirs = '../new_ba_100k/new_ba_100k/'
        else:
            seed_dirs = f'{folder_main}/aux_hyb/'
        sw = pd.read_csv(f'{folder_main}/aux_hyb/ba_test_files.csv').values
    if topology=='sw':
        suff_m='sw100k'
        #suff='sw100k_10'
        if search_full:
            seed_dirs = '../new_sw_100k/new_sw_100k/'
        else:
            seed_dirs = f'{folder_main}/aux_hyb/'
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
                            topology=topology)
    
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
    
    plt.savefig(f'{folder_imgs}/run_hybrid_once.png', bbox_inches='tight')
    metadata = {
        'R2 for test samples': all_r2_Inc,
        'Day of switch for test samples': start_days,
        'Path to the figure':'',
    }
    return {"answer": f'Ran the hybrid model and uploaded the figure {11} to {11}', 
            "metadata": metadata}


# ALLOW TO INPUT PATH TO SEIR DF! redo this func as a way to get hybr.predictions
@mcp.tool()
def run_hybrid_methods(sigma: float=0.3, gamma: float=0.2, 
               perc_switch: float=0.01, stoch: int=100, 
               topology: Literal["ba", "sw"]='ba', 
              beta_pred: list[Literal['last value',
                        'expanding mean last value',
                        'median beta', 'regression beta', 
                        'lstm']]=['median beta'],
              seed_numbers: list[int]=[93,2390]) -> dict:
    """
    Runs the hybrid model: start with the network model, 
        switch to the hybrid model with given methods of beta (infection transmission rate) prediction. 
    It returns the figure of predicted time series for incidence and beta.

    Args:
        sigma (float): The duration of the latent period (where 1/sigma is period in days)
        gamma (float): The duration of the infectious period (where 1/gamma is period in days)
        perc_switch (float): Switch to the hybrid model when the fraction 
            of Infected reaches <perc_switch>.
        stoch (int): Number of hybrid model runs to calculate interval bounds.
        topology (str): Short name of the network topology. 
            'ba' - Barabasi-Albert, 'sw' - small world.
        beta_pred (list): Method for beta prediction.
            'last value' - take last known beta value,
            'expanding mean last value' - take the ,
            'median beta' - median value of beta on train samples, 
            'regression beta' - regression model trained for beta prediction, 
            'lstm' - long short term memory model trained for beta prediction.
        seed_numbers (list): Indexes of test samples.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure).
            
    """
    if topology=='ba':
        suff_m='ba100k'
        suff='ba100k_10'
        if search_full:
            seed_dirs = '../new_ba_100k/new_ba_100k/'
        else:
            seed_dirs = f'{folder_main}/aux_hyb/'
        sw = pd.read_csv(f'{folder_main}/aux_hyb/ba_test_files.csv').values
    if topology=='sw':
        suff_m='sw100k'
        suff='sw100k_10'
        if search_full:
            seed_dirs = '../new_sw_100k/new_sw_100k/'
        else:
            seed_dirs = f'{folder_main}/aux_hyb/'
        sw = pd.read_csv(f'{folder_main}/aux_hyb/test_files.csv').values
    seed_numbers = sw[::10][seed_numbers]
    #methods = ['last value','expanding mean last value',
    #           'median beta','regression beta', 'lstm']
    plot_hyb.apply_methods(seed_dirs=seed_dirs,
              seed_numbers=seed_numbers, on_incidence=True,
              switch_on_incidence=False,
              methods=beta_pred, show_fig_flag=True,
             is_filename=True, sigma=sigma, gamma=gamma, 
              perc_switch=perc_switch, stoch=stoch, 
              m_folder=f'{folder_main}/aux_hyb/',
             suff_m=suff_m, suff=suff)
    
    plt.savefig(f'{folder_imgs}/run_hybrid_methods.png', bbox_inches='tight')
    
    metadata = {
        #".": 1,
        #**artifact_metadata,
    }
    return {"answer": f"Ran the hybrid model and uploaded the figure {11} to {11}", 
            "metadata": metadata}


@mcp.tool()
def hybrid_2x2(sigma: float=0.3, gamma: float=0.2, 
               perc_switch: float=0.05, stoch: int=10, 
               topology: Literal["ba", "sw"]='ba', 
               beta_pred: list[Literal['last value',
                        'expanding mean last value',
                        'median beta', 'regression beta', 
                        'lstm']]=['regression beta', 'lstm'],
              seed_numbers: list[int]=[93,2390]) -> dict:
    """
    Runs the hybrid model: start with the network model, 
        switch to the hybrid model with given methods of beta (infection transmission rate) prediction. 
    It returns the figure 2x2 of predicted time series for incidence and beta.

    Args:
        sigma (float): The duration of the latent period (where 1/sigma is period in days) 
        gamma (float): The duration of the infectious period (where 1/gamma is period in days) 
        perc_switch (float): Switch to the hybrid model when the fraction 
            of Infected reaches <perc_switch>.
        stoch (int): Number of hybrid model runs to calculate interval bounds.
        topology (str): Short name of the network topology. 
            'ba' - Barabasi-Albert, 'sw' - small world.
        beta_pred (list): Method for beta prediction.
            'last value' - take last known beta value,
            'expanding mean last value' - take the ,
            'median beta' - median value of beta on train samples, 
            'regression beta' - regression model trained for beta prediction, 
            'lstm' - long short term memory model trained for beta prediction.
        seed_numbers (list): Indexes of test samples.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved 2x2 figure, 
            coefficient of determination for test samples, the days of switch for test samples).
    """
    
    fig, ax = plt.subplots(2,2, figsize=(10, 6))
    ax = ax.flatten()
    
    if topology=='ba':
        suff_m='ba100k'
        suff='ba100k_10'
        sw = pd.read_csv(f'{folder_main}/aux_hyb/ba_test_files.csv').values
        if search_full:
            seed_dirs = '../new_ba_100k/new_ba_100k/'
        else:
            seed_dirs = f'{folder_main}/aux_hyb/'
    if topology=='sw':
        suff_m='sw100k'
        suff='sw100k_10'
        sw = pd.read_csv(f'{folder_main}/aux_hyb/test_files.csv').values
        if search_full:
            seed_dirs = '../new_sw_100k/new_sw_100k/'
        else:
            seed_dirs = f'{folder_main}/aux_hyb/'
    
    seed_numbers = sw[::10][seed_numbers]
    
    types_start_day = ['fraq_people']
    j=0
    labs = ['(a)','(b)','(c)','(d)'][::-1]
    r2s = []
    switches=[]
    
    for k in seed_numbers:
        for method in beta_pred:
            if 'median' in method:
                model_path = f'{folder_main}/aux_hyb/{suff_m}_median_beta.csv'
            elif 'regression beta' in method:
                model_path = f'{folder_main}/aux_hyb/{suff_m}_regression_bt.joblib'
            elif 'lstm' in method:
                model_path = f'{folder_main}/aux_hyb/{suff_m}_lstm_4_001_s10'   
            else:
                model_path=''
            _, _, _, _, all_r2_Inc, _, _, all_peak,\
                _, start_days = plot_hyb.main_f(I_prediction_method='seir', 
                                count_stoch_line=stoch, 
                                beta_prediction_method=method, 
                                type_start_day=types_start_day[0], 
                                seed_numbers=[k], show_fig_flag=True,
                                seed_dirs=seed_dirs, sigma=sigma, gamma=gamma, 
                                ax=[ax[j]], model_path=model_path,perc_switch=perc_switch,
                                is_filename=True,on_incidence=True,
                            switch_on_incidence=False)
            r2s.append(all_r2_Inc[0])
            switches.append(start_days[0])
            ax[j].text(-0.1, 1.1, labs.pop(),
                       transform=ax[j].transAxes, size=15)
            j+=1
    
    plt.savefig(f'{folder_imgs}/hybrid_2x2.png', bbox_inches='tight')
    
    metadata = {
        'R2 for test samples': r2s,
        'Days of switch for test samples': switches,
        #**artifact_metadata,
    }
    return {"answer": f'Ran the hybrid model and uploaded the figure {11} to {11}', 
            "metadata": metadata}


# --------- surrogate ----------
@mcp.tool()
def surrogate_point_2x2(topology:Literal["ba","sw"]='ba',
                          test_indices:list[int] = [11,7,1,15],
                    ) -> dict:
    """
    Runs the surrogate model: use the trained autoencoder model to output incidence time series. 
    It returns the figure 2x2 of predicted time series for incidence and beta.

    Args:
        topology (str): Short name of the network topology. 
            'ba' - Barabasi-Albert, 'sw' - small world.
        test_indices (list): Indexes of test samples.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved 2x2 figure, 
            coefficient of determination for test samples, 
            alpha (fraction of non-immune individuals) and beta (infection transmission rate) for chosen test samples).
            
    """
    type_df = 'point'
    ae = torch.load(f'{folder_main}/calibr/models/autoencoder_{topology}_100k_n.pt', 
                    weights_only=False)
    X_train, y_train, X_test, y_test,tmax = \
        surr_funcs.get_splits_df(folder=folder_main+'calibr/', 
                                 type_df=type_df, network_type = topology)
    fontsize = 12
    rows, cols = 2, 2
    fig, ax = plt.subplots(rows, cols, figsize=(10, 8))
    labels = ['(a)', '(b)', '(c)', '(d)']
    counter = 0
    cut = 100
    beta_alpha, r2s = [], []
    for row in range(rows):
        for col in range(cols):
            surrogate_sim = surr_funcs.predict(ae, 
                                               X_test[test_indices[counter]]
                                              ).numpy()
            r2 = r2_score(y_test[test_indices[counter]], surrogate_sim)
            #print(labels[counter], test_indices[counter], X_test[test_indices[counter]])
            r2s.append(r2)
            beta_alpha.append(X_test[test_indices[counter]])
            
            ax[row][col].plot(y_test[test_indices[counter]][:cut], 
                              label='Network model', marker='o', 
                              color='OrangeRed')
            ax[row][col].plot(surrogate_sim[:cut], lw=3, color='RoyalBlue', 
                              label=f'Surrogate model\n$R^2=${r2:.3f}')
    
            ax[row][col].set_xlabel('Time, days', fontsize=1.2*fontsize)
            ax[row][col].set_ylabel('Incidence, cases', fontsize=1.2*fontsize)
            ax[row][col].set_ylim(0, 3000)
            ax[row][col].set_xlim(-5, 100)
            ax[row][col].tick_params(axis='both', which='major', 
                                     labelsize=fontsize)
            ax[row][col].legend(fontsize=1.2*fontsize)
            ax[row][col].grid()
            
            # Add subplot label outside the top-left corner
            ax[row][col].text(-0.1, 1.1, labels[counter],
                        transform=ax[row][col].transAxes, size=fontsize*1.5)
            counter += 1
    
    plt.tight_layout()
    plt.savefig(f'{folder_imgs}/surrogate_point_2x2.png', bbox_inches='tight')
    metadata = {
        'R2 for test samples': r2s,
        'Alpha of test samples': list(np.array(beta_alpha)[:,1]),
        'Beta of test samples': list(np.array(beta_alpha)[:,0]),
    }
    print(metadata)
    return {"answer": f'Uploaded the figure {11} to {11}', 
            "metadata": metadata}


@mcp.tool()
def surrogate_interval_2x2(topology:Literal["ba","sw"]='ba',
                          test_indices:list[int] = [1101,7,1,150],
                    ) -> dict:
    type_df = 'interval'
    ae = torch.load(folder_main+f'/calibr/models/autoencoder_interval_{topology}_100k_n.pt', 
                    weights_only=False)
    
    X_train, y_train, X_test, y_test, tmax, mtest,qw = \
                    surr_funcs.get_splits_df(folder=folder_main+'calibr/', 
                                             folder_all=folder_main+'aux_hyb/',
                                             type_df=type_df,
                                             network_type = topology,
                                             with_orig_X=True,
                                            search_full=search_full)

    fontsize = 11
    rows, cols = 2, 2
    fig, ax = plt.subplots(rows, cols, figsize=(10, 8))
    
    labels = ['(a)', '(b)', '(c)', '(d)']
    counter = 0
    mean_index = range(tmax)
    low_index = range(tmax, 2*tmax)
    high_index = range(2*tmax, 3*tmax)
    cut = 100
    r2s, beta_alpha = [], []
    for row in range(rows):
        for col in range(cols):
            surrogate_sim = surr_funcs.predict(ae, X_test[test_indices[counter]]).numpy()
            r2 = r2_score(y_test[test_indices[counter]], surrogate_sim)
            r2s.append(r2)
            b,a = X_test[test_indices[counter]]
            beta_alpha.append([b,a])
        
            gt = np.array(y_test[test_indices[counter]])
            
            ax[row][col].plot(gt[mean_index][:cut], label='Network model, mean', 
                              marker='', color='OrangeRed')
            if search_full:
                real_idx = mtest.iloc[test_indices[counter]].name
                part = qw[qw.group==real_idx].iloc[:,5:-1]
            else:
                part = qw[(qw.beta.round(2)==round(b,2))&(
                            qw.alpha.round(2)==round(a,2))
                        ].iloc[:,5:-1]
            part.columns = part.columns.astype(int)
            ax[row][col].plot(part.T, color='OrangeRed', ls=':', alpha=.5,
                             label=['Network model, trajectory']+['']*9)
        
            ax[row][col].fill_between(np.linspace(0, tmax, tmax)[:cut], 
                                      gt[low_index][:cut], gt[high_index][:cut],
                                alpha = 0.3, color='OrangeRed', 
                                      label='Network model interval')
            
            ax[row][col].plot(surrogate_sim[mean_index][:cut], lw=2, 
                              color='RoyalBlue', label='Surrogate model, mean')
            ax[row][col].fill_between(np.linspace(0, tmax, tmax)[:cut], 
                        surrogate_sim[low_index][:cut], 
                        surrogate_sim[high_index][:cut],
                        alpha = 0.3, color='RoyalBlue', 
                                      label='Surrogate model interval')
    
            ax[row][col].set_xlabel('Time, days', fontsize=1.2*fontsize)
            ax[row][col].set_ylabel('Incidence, cases', fontsize=1.2*fontsize)
            ax[row][col].set_ylim(0, 2800)
            ax[row][col].set_xlim(-5, cut)
            ax[row][col].tick_params(axis='both', which='major', labelsize=fontsize)
            ax[row][col].legend(fontsize=fontsize)
            ax[row][col].grid()
            
            # Add subplot label outside the top-left corner
            ax[row][col].text(-0.1, 1.1, labels[counter],
                        transform=ax[row][col].transAxes, size=fontsize*1.5)
            
            counter += 1
    
    plt.tight_layout()
    plt.savefig(f'{folder_imgs}/surrogate_interval_2x2.png', bbox_inches='tight')
    metadata = {
        'R2 for test samples': r2s,
        'Alpha of test samples': list(np.array(beta_alpha)[:,1]),
        'Beta of test samples': list(np.array(beta_alpha)[:,0]),
        #**artifact_metadata,
    }
    return {"answer": f'Uploaded the figure {11} to {11}', 
            "metadata": metadata}
    

@mcp.tool()
def surrogate_heatmap_r2(topology:Literal["ba","sw"]='ba',
                    ) -> dict:
    type_df = 'point'
    ae = torch.load(f'{folder_main}/calibr/models/autoencoder_{topology}_100k_n.pt', 
                    weights_only=False)
    X_train, y_train, X_test, y_test,tmax = \
        surr_funcs.get_splits_df(folder=folder_main+'calibr/', 
                                 type_df=type_df,network_type = topology)
    dd = surr_funcs.df_for_heatmap(ae, type_df,X_train, y_train, 
                                   X_test, y_test, tmax)
    
    type_df = 'interval'
    ae = torch.load(f'{folder_main}/calibr/models/autoencoder_interval_{topology}_100k_n.pt', 
                    weights_only=False)
    X_train, y_train, X_test, y_test,tmax = \
        surr_funcs.get_splits_df(folder=folder_main+'calibr/', 
                                 type_df=type_df,network_type = topology)
    dd2_mean,dd2_min,dd2_high = surr_funcs.df_for_heatmap(ae, type_df, 
                                                          X_train, y_train, 
                                                          X_test, y_test, tmax)
    fontsize = 15
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax=axes.flatten()
    
    n = ['(a)','(b)','(c)','(d)'][::-1]
    
    cmap = surr_funcs.nonlinear_cmap()
    for i,heat_df, title in zip(range(4), 
                                [dd,dd2_mean,dd2_min,dd2_high], 
                                ['Point estimation','Interval estimation (mean)',
                                'Interval estimation (lower bound)',
                                'Interval estimation (upper bound)'
                                ]):
        
        ax_i = sns.heatmap(heat_df.sort_index(level=1, ascending=False), 
                           cmap=cmap, ax=ax[i], #norm=norm, 
                           cbar_kws={'extendfrac': .1,
                                    #"ticks":ticks, "boundaries":boundaries
                                    },
                           vmin=0, vmax=1,
                          xticklabels = 10, yticklabels=10,
                          linewidths=0.0, rasterized=True,)
        ax_i.set_title(title, fontsize=1.2*fontsize)
        ax_i.text(-0.1, 1.1, n.pop(),
                  transform=ax_i.transAxes, size=1.5*fontsize)
        ax_i.collections[0].cmap.set_bad('0.7')
        ax_i.set_xlabel(r'$\beta_n$', fontsize=1.2*fontsize)
        ax_i.set_ylabel(r'$\alpha$', fontsize=1.2*fontsize)
        ax_i.tick_params(axis='both', which='major', labelsize=fontsize)
        cbar = ax_i.collections[0].colorbar
        cbar.set_label(r'$R^2$', rotation=0, size=fontsize)
            
    for i in [-1,-2,-3,-4]:    
        ax_i.figure.axes[i].tick_params(labelsize=fontsize)
    
    #ax_1.figure.axes[-1].set_ylabel(r'$R^2$', size=fontsize)
    #ax_1.figure.axes[-2].set_ylabel(r'$R^2$', size=fontsize)
    
    plt.tight_layout()
    plt.savefig(f'{folder_imgs}/surrogate_heatmap_r2.png', bbox_inches='tight')
    metadata = {
        #**artifact_metadata,
    }
    return {"answer": f'Uploaded the figure {11} to {11}', 
            "metadata": metadata}


# --------- calibrations/forecasts ----------
@mcp.tool()
def calibrate_model_complete_data(model_name:Literal["hybrid", "network",'surrogate']='hybrid',
                    n_network_runs:int=1, show_surr_nth_line:int=5,
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
               n_hyb_runs=n_network_runs, nth=show_surr_nth_line)
    plt.savefig(f'{folder_imgs}/calibrate_model_complete_data.png', bbox_inches='tight')

    metadata = {
        "beta_mode":beta_mode,
        "alpha_mode":alpha_mode,
        #**artifact_metadata,
    }
    return {"answer": f'Calibrated the {model_name} model and uploaded the figure {11} to {11}', 
            "metadata": metadata}


@mcp.tool()
def calibrate_model_complete_data_3in1(n_network_runs:int=1, show_surr_nth_line:int=5,
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
    for gs_0i, gs_1i, idata_i,idx in zip([gs00,gs01,gs02],
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
                       network_params, pred=False,nth=show_surr_nth_line,
                              n_hyb_runs=n_network_runs,
                             ax_curves=[ax_curves],
                             ax_kde=[ax_up,ax_scatter,ax_right])
        fontsize=14
        ax_curves.annotate(labels[idx], xy=(0, 0), xycoords='axes fraction',
                               xytext=(-30, -50), textcoords='offset points',
                               fontsize=1.5*fontsize, ha='right', va='baseline')
        beta_modes.append(beta_mode)
        alpha_modes.append(alpha_mode)
        
    plt.savefig(f'{folder_imgs}/calibrate_model_complete_data_3in1.png', bbox_inches='tight')
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
                    show_surr_nth_line:int=5,
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
                       network_params, pred=True, nth = show_surr_nth_line)
    
    plt.savefig(f'{folder_imgs}/calibrate_model_forecast.png', bbox_inches='tight')
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
                   show_surr_nth_line:int=5,
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
                              nth=show_surr_nth_line,
                             ax_curves=[ax_curves],
                             ax_kde=[ax_up,ax_scatter,ax_right])
        beta_modes.append(beta_mode)
        alpha_modes.append(alpha_mode)
        fontsize=14
        ax_curves.annotate(labels[idx], xy=(0, 0), xycoords='axes fraction',
                               xytext=(-30, -50), textcoords='offset points',
                               fontsize=1.5*fontsize, ha='right', va='baseline')
        
    plt.savefig(f'{folder_imgs}/calibrate_model_forecast_3in1.png', bbox_inches='tight')
    metadata = {
        "beta_modes":beta_modes,
        "alpha_modes":alpha_modes,
        #**artifact_metadata,
    }
    return {"answer": f'Forecasted with the {model_name} model and uploaded the figure {11} to {11}', 
            "metadata": metadata}


# --------- heatmaps/aux.figs ----------
@mcp.tool()
def plot_synth_peaks() -> dict:
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    ax = axes.flatten()
    
    heat_orig = aux_f.heatmap_orig_peaks(topology='ba',
                                         folder=f'{folder_main}/aux_hyb')
    aux_f.peaks_hmaps(heat_orig, with_inc=True, 
                      title=', Barabasi-Albert', 
                      ax=[ax[0],ax[2]], n=['(a)','(c)'])
    heat_orig = aux_f.heatmap_orig_peaks(topology='sw',
                                         folder=f'{folder_main}/aux_hyb')
    aux_f.peaks_hmaps(heat_orig, with_inc=True, 
                      title=', small world', 
                      ax=[ax[1],ax[3]], n=['(b)','(d)'])
    plt.savefig(f'{folder_imgs}/plot_synth_peaks.png', bbox_inches='tight')
    metadata = {
        #**artifact_metadata,
    }
    return {"answer": f'Uploaded the figures {11} and {11} to {11}', 
            "metadata": metadata}
    

@mcp.tool()
def plot_synth_inc_beta() -> dict:
    aux_f.plot_synth_inc_beta(folder=f'{folder_main}/aux_hyb/',
                             save_folder=folder_imgs)
    metadata = {
        #**artifact_metadata,
    }
    
    return {"answer": f'Uploaded the figures {11} and {11} to {11}', 
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
                               )#.iloc[:100]
    if topology=='ba':
        observed_data = observed_data.iloc[:100]
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
    plt.savefig(f'{folder_imgs}/plot_forecast_peak_errors.png', bbox_inches='tight')
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
    plt.savefig(f'{folder_imgs}/plot_heatmap_r2.png', bbox_inches='tight')
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
    plt.savefig(f'{folder_imgs}/plot_heatmap_switch.png', bbox_inches='tight')
    metadata = {
        #**artifact_metadata,
    }
    return {"answer": f'Uploaded the figure {11} to {11}', 
            "metadata": metadata}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7331, path="/mcp")