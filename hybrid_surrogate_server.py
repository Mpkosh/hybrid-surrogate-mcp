#!/usr/bin/env python3
#fastmcp pymc arviz matplotlib numpy pandas scikit-learn scipy seaborn tensorflow torch
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
from typing import Literal
import math
import os
#from io import BytesIO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Hybrid and Surrogate Server")
folder_main = 'hybrid_surr/'
folder_imgs = 'imgs/'

'''
For demonstration purposes without large files: search_full=False.
After creating datasets and training, inference requires search_full=True and:
- hybrid_surr\num_exp\ba_incidence_100k.csv (or sw_incidence_100k.csv) 
    for .run_surrogate_interval() to plot real interval bounds.
- hybrid_surr\num_exp\*.csv as seir dataframes for .run_hybrid_model().
'''
search_full=False 


# --------- hybrid ----------
@mcp.tool()
def run_hybrid_model(sigma: float=0.3, gamma: float=0.2, 
               switch_I_fraction: float=0.05, n_hybrid_runs: int=20, 
               topology: Literal["ba", "sw"]='ba', 
               beta_pred: list[Literal['last value',
                        'expanding mean last value',
                        'median beta', 'regression beta', 
                        'lstm']]=['regression beta', 'lstm'],
              seir_df_paths: list[str]=\
               ['hybrid_surr/num_exp/p_0.13_0.3_0.2_0.0001_0.39_seed_0.csv',
                'hybrid_surr//num_exp/p_0.99_0.3_0.2_0.0001_0.63_seed_0.csv'],
              save_results:bool = False,
              res_folder_name:str='example',
              show_plots: bool = True,
              ) -> dict:
    """
    The tool runs the hybrid model: start with the network model to model SEIR
    compartments, switch from the network to the hybrid model with given methods
    of beta (infection transmission rate) prediction. Can save results: RMSE/R^2,
    prediction time, day of switch, peak time and height.
    
    Args:
        sigma (float): The duration of the latent period (where 1/sigma is the
            period in days)
        gamma (float): The duration of the infectious period (where 1/gamma is
            the period in days)
        switch_I_fraction (float): Switch from the network to the hybrid model
            when the fraction of Infected reaches <perc_switch>.
        n_hybrid_runs (int): Number of hybrid model runs to calculate interval
            bounds.
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
        beta_pred (list): Method for beta prediction.
            - 'last value': take last known beta value,
            - 'expanding mean last value': take the cumulative average of beta
            values,
            - 'median beta': median value of beta on train samples,
            - 'regression beta': regression model trained for beta prediction,
            - 'lstm': long short term memory model trained for beta prediction.
        seir_df_paths (list): paths to dataframes containing generated
            network-model trajectories with SEIR compartments.
        save_results (bool): Whether to save hybrid model predictions and metrics
            in a new dataframe.
        res_folder_name (str): Short folder name to save results to.
        show_plots (bool): Whether to show plots with hybrid model predictions.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure of predicted time
            series for incidence and beta, coefficient of determination for test
            samples, the days of switch for test samples).
    """

    n_unique = len(seir_df_paths)*len(beta_pred)
    
    if show_plots:
        rows = n_unique//2+math.ceil(n_unique%2)
        fig, ax = plt.subplots(rows, 2, figsize=(10, 3*rows))
        ax = ax.flatten()
    else:
        ax = [0]*n_unique
        
    if topology=='ba':
        suff_m='ba100k'
        suff=suff_m+f'sI_fullR_{int(switch_I_fraction*100)}'
    if topology=='sw':
        suff_m='sw100k'
        suff=suff_m+f'sI_fullR_{int(switch_I_fraction*100)}'
        
    seed_dirs = ''
    
    types_start_day = ['fraq_people']
    type_start_day=types_start_day[0] # for now
    j=0
    
    method_label_d = {'last value': 'last_value',
                 'expanding mean last value': 'expanding_mean_last_value',
                 'median beta': 'median_beta',
                 'regression beta': 'regression_beta',
                 'lstm': 'lstm_day_E_previous_I'}
    

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    if len(alphabet)< n_unique:
        alphabet += 'X'*(n_unique-len(alphabet))
    labs = ['('+alphabet[index] + ')' for index in range(n_unique)
           ][::-1]
    r2s = []
    switches=[]
    m_folder = f'{folder_main}/num_exp/'
    
    for k in seir_df_paths:
        for method in beta_pred:
            if 'median' in method:
                model_path = f'{folder_main}/num_exp/{suff_m}_median_beta.csv'
            elif 'regression beta' in method:
                model_path = f'{folder_main}/num_exp/{suff_m}_regression_bt.joblib'
            elif 'lstm' in method:
                model_path = f'{folder_main}/num_exp/{suff_m}_lstm_4_001_s10'   
            else:
                model_path=''
            try:
                all_rmse_I, all_rmse_Inc, all_rmse_Beta, \
                    all_r2, all_r2_Inc, all_r2_full, all_r2_Inc_full,\
                    all_peak, execution_time, start_days = \
                        plot_hyb.main_f(I_prediction_method='seir', 
                                    count_stoch_line=n_hybrid_runs, 
                                    beta_prediction_method=method, 
                                    type_start_day=types_start_day[0], 
                                    seed_numbers=[k], show_fig_flag=show_plots,
                                    seed_dirs=seed_dirs, sigma=sigma, gamma=gamma, 
                                    ax=[ax[j]], model_path=model_path,
                                    perc_switch=switch_I_fraction,
                                    is_filename=False,on_incidence=True,
                                    switch_on_incidence=False,
                                    topology=topology,
                                    res_folder_name=res_folder_name)
                r2s.append(all_r2_Inc)
                switches.append(start_days)
                if show_plots:
                    ax[j].text(-0.1, 1.1, labs.pop(),
                               transform=ax[j].transAxes, size=15)
                j+=1
    
                if save_results:
                    # creating a dataframe for peaks
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
                        f'{type_start_day}': start_days})
    
                    # merging dataframes
                    results = pd.concat([rmse_df, all_peak], axis=1)
                    folder_name = res_folder_name#seed_numbers.split('/')[0]

            except FileNotFoundError as e:
                pass
                #print(e)
                
            if save_results:
                path = f'{m_folder}/results/{folder_name}/{type_start_day}/'
                if not os.path.exists(path):
                    os.makedirs(path)
                new_label=method_label_d.get(method,'q')
                results.to_csv(f'{path}/{new_label}_results_{suff}.csv', 
                           index=False)
                
    answer_plots, answer_results = '','',
    fig_path, res_path = '',''
    if show_plots:
        fig_path = f'{folder_imgs}/run_hybrid_model.png'
        plt.savefig(fig_path, bbox_inches='tight')
        answer_plots = ', saved the plots for the hybrid model'
    if save_results:
        answer_results = ', saved the predicted trajectories'
        res_path = f'{folder_main}/num_exp/'+'/results/'+res_folder_name
        
    r2s = np.array(r2s).flatten().tolist()
    switches = np.array(switches).flatten().tolist()
    metadata = {
        'R2 for test samples': r2s,
        'Days of switch for test samples': switches,
        'Path to the saved figure': fig_path,
        'Path to the saved results': res_path,
    }
    return {"answer": f'Ran the hybrid model{answer_plots}{answer_results}.', 
            "metadata": metadata}
    


@mcp.tool()
def hybrid_heatmap_r2(topology:Literal["ba","sw"]='ba',
                    ) -> dict:
    """
    The tool creates several heatmaps with values of coefficient of determination
    by running the hybrid model for each alpha (initial fraction of
    non-immune individuals) and beta (infection transmission rate)
    pair of test samples.
    
    Args:
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved heatmap with values of
            coefficient of determination).
    """
    switch_perc = 5
    metric = 'r2'
    switch='fraq_people'
    fin_inc = aux_f.df_metrics(folder_name=f'{folder_main}/num_exp/',
                        top_name=f'new_{topology}_100000', 
                       test_suff=f'{topology}_', 
                       switch=switch,
                       with_inc=True, 
                       trim=False, suff=f'_{topology}100k_sI_fullR_{switch_perc}')
    
    aux_f.metric_hmaps(folder_name=f'{folder_main}/num_exp/',
                       fin=fin_inc, met=f'{metric}_Inc', 
                       suff=f'_{topology}_sI_{switch_perc}', 
                       exclude=['Cumulative Average',
                                'Exponential Decay'],
                        save=False)
    
    fig_path = f'{folder_imgs}/hybrid_heatmap_r2.png'
    plt.savefig(fig_path, bbox_inches='tight')
    metadata = {
        'Path to the saved figure': fig_path
    }
    return {"answer": 'Created the heatmap plot for the hybrid model.',
            "metadata": metadata}


# --------- surrogate ----------
@mcp.tool()
def run_surrogate_point(topology:Literal["ba","sw"]='ba',
                        alphas:list[float] = [0.44, 0.59, 0.71, 0.75],
                        betas:list[float] = [0.35, 0.4 , 0.41, 0.37],
                    ) -> dict:
    """
    The tool runs the surrogate model with point estimation: for each 
    alpha (initial fraction of non-immune individuals) and beta (infection 
    transmission rate) pair, the model outputs incidence values.
    
    Args:
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
        alphas (list): A list of alpha values for each surrogate model
            simulation.
        betas (list): A list of beta values for each surrogate model
            simulation.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure of predicted time
            series, coefficient of determination for each sample).
    """
    
    type_df = 'point'
    ae = torch.load(f'{folder_main}/num_exp/models/autoencoder_{topology}_100k_n.pt', 
                    weights_only=False)
    df = pd.read_csv(folder_main+'/num_exp/'+f'/{topology}_{type_df}_dataset.csv', 
                     index_col=0)
    df[['beta','alpha']] = df[['beta','alpha']].round(2)
    
    fontsize = 12
    rows = len(alphas)//2+math.ceil(len(alphas)%2)
    
    fig, ax = plt.subplots(rows, 2, figsize=(10, 4*rows))
    ax = ax.flatten()
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    if len(alphabet)< len(alphas):
        alphabet += 'X'*(len(alphas)-len(alphabet))
    labels = ['('+alphabet[index] + ')' for index in range(len(alphas))]
    j = 0
    cut = 100

    r2s = []
    for ax_i,beta,alpha in zip(ax, betas, alphas):
        surrogate_sim = surr_funcs.predict(ae, 
                                           [beta,alpha]
                                          ).numpy()
        real_inc = df[(df.beta==beta)&(df.alpha==alpha)
                     ].iloc[:,5:].values.flatten().tolist()
        r2 = r2_score(real_inc, surrogate_sim)
        r2s.append(r2)

        ax_i.plot(real_inc[:cut], 
                          label='Network model', marker='o', 
                          color='OrangeRed')
        ax_i.plot(surrogate_sim[:cut], lw=3, color='RoyalBlue', 
                          label=f'Surrogate model\n$R^2=${r2:.3f}')

        ax_i.set_xlabel('Time, days', fontsize=1.2*fontsize)
        ax_i.set_ylabel('Incidence, cases', fontsize=1.2*fontsize)
        ax_i.set_ylim(0, 3000)
        ax_i.set_xlim(-5, 100)
        ax_i.tick_params(axis='both', which='major', 
                                 labelsize=fontsize)
        ax_i.legend(fontsize=1.2*fontsize)
        ax_i.grid()

        # Add subplot label outside the top-left corner
        ax_i.text(-0.1, 1.1, labels[j],
                    transform=ax_i.transAxes, size=fontsize*1.5)
        j += 1
    
    plt.tight_layout()
    fig_path=f'{folder_imgs}/surrogate_point.png'
    plt.savefig(fig_path, bbox_inches='tight')
    
    r2s = np.array(r2s).flatten().tolist()
    metadata = {
        'R2 for test samples': r2s,
        'Path to the saved figure':fig_path
    }
    return {"answer": 'Created a 2x2 plot for the surrogate model'+\
                ' with point estimation.', 
            "metadata": metadata}


@mcp.tool()
def run_surrogate_interval(topology:Literal["ba","sw"]='ba',
                          alphas:list[float] = [0.32, 0.59, 0.71, 0.6 ],
                        betas:list[float] = [0.83, 0.4 , 0.41, 0.66],
                    ) -> dict:
    """
    The tool runs the surrogate model with interval estimation: for each 
    alpha (initial fraction of non-immune individuals) and beta (infection 
    transmission rate) pair, the model outputs lower estimate, mean, and 
    higher estimate of incidence values.
    
    Args:
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
        alphas (list): A list of alpha values for each surrogate model
            simulation.
        betas (list): A list of beta values for each surrogate model
            simulation.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure of predicted time
            series for incidence, coefficient of determination for each sample).
    """
    
    type_df = 'interval'
    ae = torch.load(folder_main+\
                    f'/num_exp/models/autoencoder_interval_{topology}_100k_n.pt', 
                        weights_only=False)
    if search_full:
        X_train, y_train, X_test, y_test, tmax, mtest,df_stoch_ts = \
                    surr_funcs.get_splits_df(folder=folder_main+'calibr/', 
                                             folder_all=folder_main+'num_exp/',
                                             type_df=type_df,
                                             network_type = topology,
                                             with_orig_X=True,
                                            search_full=search_full)
    else:
        df_stoch_ts = pd.read_csv(folder_main+\
                                  '/num_exp/'+f'/{topology}_4id_10samples.csv')
    df_stoch_ts[['beta','alpha']] = df_stoch_ts[['beta','alpha']].round(2)

    #type_df = 'point'
    df_mean_ts = pd.read_csv(folder_main+\
                             '/num_exp/'+f'/{topology}_{type_df}_dataset.csv', 
                     index_col=0)
    df_mean_ts[['beta','alpha']] = df_mean_ts[['beta','alpha']].round(2)

    fontsize = 11
    rows = len(alphas)//2+math.ceil(len(alphas)%2)
    fig, ax = plt.subplots(rows, 2, figsize=(10, 4*rows))
    ax = ax.flatten()
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    if len(alphabet)< len(alphas):
        alphabet += 'X'*(len(alphas)-len(alphabet))
    labels = ['('+alphabet[index] + ')' for index in range(len(alphas))]

    if topology == 'ba':
        tmax = 150
        cut = 100
    elif topology == 'sw':
        tmax = 350
        cut = 350
    mean_index = range(tmax)
    low_index = range(tmax, 2*tmax)
    high_index = range(2*tmax, 3*tmax)
    j = 0


    r2s = []
    for ax_i,beta,alpha in zip(ax, betas, alphas):
        surrogate_sim = surr_funcs.predict(ae, 
                                           [beta,alpha]
                                          ).numpy()
        real_inc = df_mean_ts[(df_mean_ts.beta==beta)&(df_mean_ts.alpha==alpha)
                     ].iloc[:,5:].values.flatten()#.tolist()
        r2 = r2_score(real_inc, surrogate_sim)
        r2s.append(r2)

        
        part = df_stoch_ts[(df_stoch_ts.beta==beta)&(
                        df_stoch_ts.alpha==alpha)
                    ].iloc[:,5:-1]
        part.columns = part.columns.astype(int)
        ax_i.plot(part.T, color='OrangeRed', ls=':', alpha=.5,
                         label=['Network model, trajectory']+['']*9)
        ax_i.plot(real_inc[mean_index][:cut], label='Network model, mean', 
                                  marker='', color='OrangeRed')
        ax_i.fill_between(np.linspace(0, tmax, tmax)[:cut], 
                           real_inc[low_index][:cut],
                          real_inc[high_index][:cut],
                            alpha = 0.3, color='OrangeRed', 
                                  label='Network model interval')

        #ax_i.plot(surrogate_sim[:cut], lw=3, color='RoyalBlue', 
        #                  label=f'Surrogate model\n$R^2=${r2:.3f}')
        ax_i.plot(surrogate_sim[mean_index][:cut], lw=2, 
                                  color='RoyalBlue', 
                                  label='Surrogate model, mean')
        ax_i.fill_between(np.linspace(0, tmax, tmax)[:cut], 
                            surrogate_sim[low_index][:cut], 
                            surrogate_sim[high_index][:cut],
                            alpha = 0.3, color='RoyalBlue', 
                                          label='Surrogate model interval')

        ax_i.set_xlabel('Time, days', fontsize=1.2*fontsize)
        ax_i.set_ylabel('Incidence, cases', fontsize=1.2*fontsize)
        ax_i.set_ylim(0, 2800)
        ax_i.set_xlim(-5, cut)
        ax_i.tick_params(axis='both', which='major', 
                                 labelsize=fontsize)
        ax_i.legend(fontsize=fontsize)
        ax_i.grid()

        # Add subplot label outside the top-left corner
        ax_i.text(-0.1, 1.1, labels[j],
                    transform=ax_i.transAxes, size=fontsize*1.5)
        j += 1
    plt.tight_layout()
    
    fig_path=f'{folder_imgs}/surrogate_interval.png'
    plt.savefig(fig_path, bbox_inches='tight')
    r2s = np.array(r2s).flatten().tolist()
    metadata = {
        'R2 for test samples': r2s,
        'Path to the saved figure': fig_path
    }
    return {"answer": 'Created the 2x2 plot for the surrogate model '+\
                'with interval estimation.', 
            "metadata": metadata}
    

@mcp.tool()
def surrogate_heatmap_r2(topology:Literal["ba","sw"]='ba',
                    ) -> dict:
    """
    The tool creates several heatmaps with values of coefficient of determination
    by running the surrogate model with point and interval estimation
    for each alpha (initial fraction of non-immune individuals)
    and beta (infection transmission rate) pair of test samples.
    
    Args:
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved heatmap with values of
            coefficient of determination).
    """
        
    type_df = 'point'
    ae = torch.load(f'{folder_main}/num_exp/models/autoencoder_{topology}_100k_n.pt', 
                    weights_only=False)
    X_train, y_train, X_test, y_test,tmax = \
        surr_funcs.get_splits_df(folder=folder_main+'num_exp/', 
                                 type_df=type_df,network_type = topology)
    dd = surr_funcs.df_for_heatmap(ae, type_df,X_train, y_train, 
                                   X_test, y_test, tmax)
    
    type_df = 'interval'
    ae = torch.load(f'{folder_main}/num_exp/models/autoencoder_interval_{topology}_100k_n.pt', 
                    weights_only=False)
    X_train, y_train, X_test, y_test,tmax = \
        surr_funcs.get_splits_df(folder=folder_main+'num_exp/', 
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
    fig_path = f'{folder_imgs}/surrogate_heatmap_r2.png'
    plt.savefig(fig_path, bbox_inches='tight')
    metadata = {
        'Path to the saved figure': fig_path,
    }
    return {"answer": 'Created the heatmap plot for the surrogate model.', 
            "metadata": metadata}


# --------- calibrations/forecasts ----------
@mcp.tool()
def calibrate_model_complete_data(model_name:Literal["hybrid", "network",'surrogate']='hybrid',
                    n_network_runs:int=1, show_surr_nth_line:int=5,
                    sigma: float=0.3, gamma: float=0.2,
                    true_alpha:float = 0.95,
                    true_beta:float = 0.1,
                    switch_I_fraction:int=0.05,
                    n_nodes:int=100000,
                    topology:Literal["ba","sw"]='ba') -> dict:
    """
    The tool calibrates parameters of a chosen model to a target incidence curve
    simulated by the network model. The calibration employs Approximate Bayesian
    Computation with Sequential Monte Carlo (ABC-SMC).
    
    Args:
        model_name (str): The model to use.
            - "hybrid", 
            - "network", 
            - "surrogate".
        n_network_runs (int): the number of times to run the network model
            (relevant only if model_name=="hybrid"/"network").
        show_surr_nth_line (int): show each <show_surr_nth_line> line on plots
            to speed up data plotting (relevant only if
            model_name=="surrogate").
        sigma (float): The duration of the latent period (where 1/sigma is
            period in days).
        gamma (float): The duration of the infectious period (where 1/gamma is
            period in days).
        true_alpha (float): alpha (initial fraction of non-immune individuals)
            of a target incidence trajectory.
        true_beta (float): beta (infection transmission rate) of a target
            incidence trajectory.
        switch_I_fraction (float): Switch from the network to the hybrid model
            when the fraction of Infected reaches <perc_switch> (relevant only
            if model_name=="hybrid").
        n_nodes (int): The number of nodes for initializing the network model
            (relevant only if model_name=="hybrid"/"network").
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure with posterior-sampled
            trajectories and posterior parameter distributions, best 
            coefficient of determination, the selected alpha value, 
            the selected beta value).
    """
    
    folder=f'{folder_main}/calibr/'
    
    if topology=='ba':
        cut = 100
    if topology=='sw':
        cut = 350
    observed_data = pd.read_csv(folder+\
                                f'observed_incidence_a{true_alpha}_b{true_beta}.csv'
                           ).iloc[:cut]
    observed_data.columns=['incidence']

    # _________
    with_switch=np.array(True) 
    num_runs=[1]
    

    if model_name=='hybrid':
        model_str = 'hyb'
    elif model_name=='network':
        model_str = 'net'
    elif model_name=='surrogate':
        model_str = 'surr'
        
    top=topology
    koeff=1
    shift=0
    network_params=[with_switch,num_runs,[switch_I_fraction],sigma,
                    gamma,n_nodes,top,koeff,shift]

    model_idata=f"{top}_{model_str}_a{true_alpha}_b{true_beta}.nc"
    idata = az.from_netcdf(folder+model_idata)
    if model_name=='surrogate':
        idata = idata.rename({"beta": "tau"})
        
    beta_mode, alpha_mode, best_r2 = plot_funcs.plot_calib(observed_data, idata, 
                                    true_beta, true_alpha, network_params, 
                                    n_hyb_runs=n_network_runs, 
                                    nth=show_surr_nth_line)

    fig_path=f'{folder_imgs}/calibrate_model_complete_data.png'
    plt.savefig(fig_path, bbox_inches='tight')

    metadata = {
        "Selected alpha value":alpha_mode,
        "Selected beta value":beta_mode,
        "Best R2":best_r2,
        'Path to the saved figure': fig_path,
    }
    return {"answer": f'Calibrated the {model_name} model and created the plot.', 
            "metadata": metadata}


@mcp.tool()
def calibrate_model_complete_data_3in1(n_network_runs:int=1, 
                                       show_surr_nth_line:int=5,
                                       sigma: float=0.3, gamma: float=0.2,
                                        true_alpha:float = 0.95,
                                        true_beta:float = 0.1,
                                        switch_I_fraction:int=0.05,
                                        n_nodes:int=100000,
                                      topology:Literal["ba","sw"]='ba') -> dict:
    """
    The tool calibrates parameters of three models (network, hybrid, surrogate)
    to a target incidence curve simulated by the network model.
    The calibration employs Approximate Bayesian Computation with
    Sequential Monte Carlo (ABC-SMC).
    
    Args:
        n_network_runs (int): the number of times to run the network model
            (relevant for the hybrid and network models).
        show_surr_nth_line (int): show each <show_surr_nth_line> line on plots
            to speed up data plotting (relevant for the surrogate model).
        sigma (float): The duration of the latent period (where 1/sigma is
            period in days).
        gamma (float): The duration of the infectious period (where 1/gamma is
            period in days).
        true_alpha (float): alpha (initial fraction of non-immune individuals)
            of a target incidence trajectory.
        true_beta (float): beta (infection transmission rate) of a target
            incidence trajectory.
        switch_I_fraction (float): Switch from the network to the hybrid model
            when the fraction of Infected reaches <perc_switch> (relevant for
            the hybrid model).
        n_nodes (int): The number of nodes for initializing the network model
            (relevant for the hybrid and network modelsfrac).
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure with posterior-sampled
            trajectories and posterior parameter distributions for the network,
            hybrid and surrogate models; selected alpha values; selected beta
            values; best coefficients of determination).
    """
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
    
    folder=f'{folder_main}/calibr/'
    if topology=='ba':
        cut = 100
    if topology=='sw':
        cut = 350
        
    observed_data = pd.read_csv(folder+\
                                f'observed_incidence_a{true_alpha}_b{true_beta}.csv'
                           ).iloc[:cut]
    observed_data.columns=['incidence']

    # _________
    with_switch=np.array(True) 
    num_runs=[1]
    top=topology
    koeff=1
    shift=0
    network_params=[with_switch,num_runs,[switch_I_fraction],sigma,
                    gamma,n_nodes,top,koeff,shift]
    idatas = [az.from_netcdf(folder+\
                    f"{top}_{model_str}_a{true_alpha}_b{true_beta}.nc") \
                  for model_str in ['net','hyb','surr']
                 ]
    # surr model's idata was saved with other arguments
    idatas[-1] = idatas[-1].rename({"beta": "tau"})
    
    beta_modes, alpha_modes, best_r2s = [],[],[]
    for gs_0i, gs_1i, idata_i,idx in zip([gs00,gs01,gs02],
                                 [gs10,gs11,gs12],
                                 idatas,
                                     np.arange(3)):
    
        if idx==1:
            with_switch=np.array(True) 
            network_params=[with_switch,num_runs,switch_I_fraction,sigma,
                    gamma,n_nodes,top,koeff,shift]
        elif idx == 2:
            num_runs = [0]
            switch_I_fraction=[1]
            with_switch=np.array(False) 
            network_params=[with_switch,num_runs,switch_I_fraction,sigma,
                    gamma,n_nodes,top,koeff,shift]
    
        ax_curves = fig.add_subplot(gs_0i[0])
        ax_up = fig.add_subplot(gs_1i[0])
        ax_scatter = fig.add_subplot(gs_1i[2], sharex=ax_up)
        ax_right = fig.add_subplot(gs_1i[3], sharey=ax_scatter)
    
        beta_mode, alpha_mode, best_r2 = plot_funcs.plot_calib(observed_data, idata_i, 
                                    true_beta, true_alpha, network_params, 
                                    pred=False,nth=show_surr_nth_line,
                                    n_hyb_runs=n_network_runs,
                                    ax_curves=[ax_curves],
                                    ax_kde=[ax_up,ax_scatter,ax_right])
        fontsize=14
        ax_curves.annotate(labels[idx], xy=(0, 0), xycoords='axes fraction',
                               xytext=(-30, -50), textcoords='offset points',
                               fontsize=1.5*fontsize, ha='right', va='baseline')
        beta_modes.append(beta_mode)
        alpha_modes.append(alpha_mode)
        best_r2s.append(best_r2)
        
    fig_path = f'{folder_imgs}/calibrate_model_complete_data_3in1.png'
    plt.savefig(fig_path,bbox_inches='tight')
    metadata = {
        "Selected alpha values":alpha_modes,
        "Selected beta values":beta_modes,
        "Best R2":best_r2s,
        'Path to the saved figure': fig_path,
    }
    return {"answer": 'Calibrated the network, hybrid and surrogate models'+\
                'and created the plot.', 
            "metadata": metadata}
    

@mcp.tool()
def calibrate_model_forecast(model_name:Literal["hybrid",
                             'surrogate']='hybrid',
                    show_surr_nth_line:int=5,
                    start_forecasting:Literal["14b","7b","7a"]='14b',
                    sigma: float=0.3, gamma: float=0.2,
                    true_alpha:float = 0.95,
                    true_beta:float = 0.1,
                    switch_I_fraction:int=0.05,
                    n_nodes:int=100000,
                    topology:Literal["ba","sw"]='ba'
                              ) -> dict:
    """
    The tool conducts short-time forecasting of disease incidence.
    It calibrates parameters of a chosen model to an incomplete target
    incidence curve simulated by the network model. The calibration employs
    Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC).
    
    Args:
        model_name (str): The model to use.
            - "hybrid", 
            - "network", 
            - "surrogate".
        show_surr_nth_line (int): Show each <show_surr_nth_line> line on plots
            to speed up data plotting (relevant only if
            model_name=="surrogate").
        start_forecasting (str): Short string describing when to cut off target
            incidence for calibration, i.e. start forecasting.
            - "14b": 14 days before peak target incidence,
            - "7b": 7 days before peak target incidence,
            - "7a": 7 days after peak target incidence.
        sigma (float): The duration of the latent period (where 1/sigma is
            period in days).
        gamma (float): The duration of the infectious period (where 1/gamma is
            period in days).
        true_alpha (float): Alpha (initial fraction of non-immune individuals)
            of a target incidence trajectory.
        true_beta (float): Beta (infection transmission rate) of a target
            incidence trajectory.
        switch_I_fraction (float): Switch from the network to the hybrid model
            when the fraction of Infected reaches <perc_switch> (relevant only
            if model_name=="hybrid").
        n_nodes (int): The number of nodes for initializing the network model
            (relevant only if model_name=="hybrid"/"network").
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure with posterior-sampled
            trajectories and posterior parameter distributions for the chosen
            model; the selected alpha value; the selected beta value).
    """
    
    folder=f'{folder_main}/calibr/'
    if topology=='ba':
        cut = 100
    if topology=='sw':
        cut = 350
    observed_data = pd.read_csv(folder+\
                                f'observed_incidence_a{true_alpha}_b{true_beta}.csv'
                           ).iloc[:cut]
    observed_data.columns=['incidence']
    
    with_switch=np.array(False) 
    num_runs=[1]
    top=topology
    koeff=1
    shift=0
    network_params=[with_switch,num_runs,[switch_I_fraction],
                    sigma,gamma,n_nodes,top,koeff,shift]
    
    if model_name=='hybrid':
        model_str = 'hyb'
    elif model_name=='surrogate':
        model_str = 'surr'
        
    model_idata=f"{top}_{model_str}_a{true_alpha}_b{true_beta}_{start_forecasting}.nc"
    idata = az.from_netcdf(folder+model_idata)

    beta_mode, alpha_mode, _ = plot_funcs.plot_calib(observed_data, idata, 
                       true_beta, true_alpha, 
                       network_params, pred=True, nth = show_surr_nth_line)
    
    fig_path = f'{folder_imgs}/calibrate_model_forecast.png'
    plt.savefig(fig_path, bbox_inches='tight')
    metadata = {
        "Selected alpha value":alpha_mode,
        "Selected beta value":beta_mode,
        'Path to the saved figure': fig_path,
    }
    return {"answer": f'Forecasted with the {model_name} model and created the plot.', 
            "metadata": metadata}


@mcp.tool()
def calibrate_model_forecast_3in1(model_name:Literal["hybrid",
                             'surrogate']='hybrid',
                   show_surr_nth_line:int=5,
                   sigma: float=0.3, gamma: float=0.2,
                   true_beta:float = 0.1,true_alpha:float = 0.95,
                    switch_I_fraction:int=0.05,
                    n_nodes:int=100000,
                   topology:Literal["ba","sw"]='ba') -> dict:
    """
    The tool conducts short-time forecasting of disease incidence.
    It calibrates parameters of the chosen model to an incomplete target
    incidence curve simulated by the network model. The calibration employs
    Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC).
    
    Args:
        model_name (str): The model to use.
            - "hybrid", 
            - "network", 
            - "surrogate".
        show_surr_nth_line (int): Show each <show_surr_nth_line> line on plots
            to speed up data plotting (relevant only if
            model_name=="surrogate").
        sigma (float): The duration of the latent period (where 1/sigma is
            period in days).
        gamma (float): The duration of the infectious period (where 1/gamma is
            period in days).
        true_alpha (float): alpha (initial fraction of non-immune individuals)
            of a target incidence trajectory.
        true_beta (float): beta (infection transmission rate)
            of a target incidence trajectory.
        switch_I_fraction (float): Switch from the network to the hybrid model
            when the fraction of Infected reaches <perc_switch> (relevant only
            if model_name=="hybrid").
        n_nodes (int): The number of nodes for initializing the network model
            (relevant only if model_name=="hybrid"/"network").
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
    
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure with posterior-sampled
            trajectories and posterior parameter distributions for the chosen
            model; selected alpha values; selected beta values).
    """
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    labels = ['('+alphabet[index] + ')' for index in range(len(alphabet))]

    folder=f'{folder_main}/calibr/'
    if topology=='ba':
        cut = 100
    if topology=='sw':
        cut = 350
    observed_data = pd.read_csv(folder+\
                                f'observed_incidence_a{true_alpha}_b{true_beta}.csv'
                           ).iloc[:cut]
    observed_data.columns=['incidence']
    
    with_switch=np.array(False) 
    num_runs=[1]
    top=topology
    koeff=1
    shift=0
    network_params=[with_switch,num_runs,[switch_I_fraction],
                    sigma,gamma,n_nodes,top,koeff,shift]
    
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
                        f"{top}_{model_str}_a{true_alpha}_b{true_beta}_{start_pred}.nc") \
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
    
        beta_mode, alpha_mode,_ = plot_funcs.plot_calib(observed_data, idata_i, 
                                               true_beta, true_alpha, 
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
    
    fig_path = f'{folder_imgs}/calibrate_model_forecast_3in1.png'
    plt.savefig(fig_path, bbox_inches='tight')
    metadata = {
        "Selected alpha values":alpha_modes,
        "Selected beta values":beta_modes,
        'Path to the saved figure': fig_path,

    }
    return {"answer": f'Forecasted with the {model_name} model and created the plot.', 
            "metadata": metadata}


# --------- heatmaps/aux.figs ----------
@mcp.tool()
def plot_synth_peaks() -> dict:
    """
    The tool creates several heatmaps showing the distribution of peak time
    and peak incidence for synthetic incidence curves for Barabási–Albert
    and small-world topologies.
    
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure).
    """
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    ax = axes.flatten()
    
    heat_orig = aux_f.heatmap_orig_peaks(topology='ba',
                                         folder=f'{folder_main}/num_exp')
    aux_f.peaks_hmaps(heat_orig, with_inc=True, 
                      title=', Barabasi-Albert', 
                      ax=[ax[0],ax[2]], n=['(a)','(c)'])
    heat_orig = aux_f.heatmap_orig_peaks(topology='sw',
                                         folder=f'{folder_main}/num_exp')
    aux_f.peaks_hmaps(heat_orig, with_inc=True, 
                      title=', small world', 
                      ax=[ax[1],ax[3]], n=['(b)','(d)'])
    
    fig_path = f'{folder_imgs}/plot_synth_peaks.png'
    plt.savefig(fig_path, bbox_inches='tight')
    metadata = {
        'Path to the saved figure': fig_path,
    }
    return {"answer": 'Created the heatmaps showing the distribution of'+\
                ' peak time and peak incidence.', 
            "metadata": metadata}
    

@mcp.tool()
def plot_synth_inc_beta() -> dict:
    """
    The tool creates plots showing incidence trajectories and beta (infection
    transmission rate) trajectories for Barabási–Albert and small-world
    topologies.
    
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (paths to the saved figures).
    """
    fig_path_inc,fig_path_beta = \
        aux_f.plot_synth_inc_beta(folder=f'{folder_main}/num_exp/',
                                  save_folder='imgs/')
    metadata = {
        'Paths to the saved figures': [fig_path_inc,fig_path_beta],
    }
    
    return {"answer": 'Created the plot showing incidence and beta trajectories.', 
            "metadata": metadata}


@mcp.tool()
def plot_forecast_peak_errors(true_alpha:float = 0.95,
                              true_beta:float = 0.1,
                              topology:Literal["ba","sw"]='ba',
                             ) -> dict:
    """
    The tool creates plots showing peak errors for hybrid and surrogate
    approaches for short-term forecasting using incidence data: 14 days before
    peak, 7 days before peak, 7 days after peak.
    
    Args:
        true_alpha (float): alpha (initial fraction of non-immune individuals)
            of a target incidence trajectory used during calibration.
        true_beta (float): beta (infection transmission rate) of a target
            incidence trajectory used during calibration.
        topology (str): Short name of the network topology.
             - 'ba': Barabasi-Albert, 
             - 'sw': small world.
    
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (paths to the saved figure).
    """
    start_forecasting = ["14b","7b","7a"]
    top=topology
    folder = f'{folder_main}/calibr/'
    if topology=='ba':
        cut = 100
    if topology=='sw':
        cut = 350
    observed_data = pd.read_csv(folder+\
                                f'observed_incidence_a{true_alpha}_b{true_beta}.csv'
                               ).iloc[:cut]
    observed_data.columns=['incidence']
    model_strs = ['hyb','surr']
    idatas = [[az.from_netcdf(folder+\
                  f"{top}_{model_str}_a{true_alpha}_b{true_beta}_{start_pred}.nc") \
                  for model_str in model_strs
                ] for start_pred in start_forecasting
             ]
    
    aux_f.create_peak_plot(folder_name=folder,
                     observed_data = observed_data,
                       idatas = idatas,
                     with_outliers=True, same_lims=True, 
                     figsize=(3.6*len(idatas),4), x_lim = (-30, 30), y_lim = (0, 1.5),
                     alpha_m=0.08, alpha_area=0.3, save=False)
    
    fig_path = f'{folder_imgs}/plot_forecast_peak_errors.png'
    plt.savefig(fig_path, bbox_inches='tight')
    
    metadata = {
       'Path to the saved figure': fig_path,
    }
    return {"answer": 'Created the plot showing peak errors for '+\
                'hybrid and surrogate approaches for short-term forecasting.', 
            "metadata": metadata}


@mcp.tool()
def plot_heatmap_switch(topology:Literal["ba","sw"]='ba',
                    ) -> dict:
    """
    The tool creates several heatmaps showing switching behavior for test
    samples: difference between epidemic peak time and day of switch;
    distribution of switch days across all runs.
    
    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure).
    """
    switch_perc = 5
    switch='fraq_people'
    fin_inc = aux_f.df_metrics(folder_name=f'{folder_main}/num_exp/',
                        top_name=f'new_{topology}_100000', 
                       test_suff=f'{topology}_', 
                       switch=switch,
                       with_inc=True, 
                       trim=False, suff=f'_{topology}100k_sI_fullR_{switch_perc}')
    
    aux_f.smth_hmaps(fin_inc, 21)
    
    fig_path = f'{folder_imgs}/plot_heatmap_switch.png'
    plt.savefig(fig_path, bbox_inches='tight')
    metadata = {
        'Path to the saved figure': fig_path,
    }
    return {"answer": 'Created the plot showing switching behavior for test samples.', 
            "metadata": metadata}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7331, path="/mcp")