import logging
import math
import os
import sys
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import \
    r2_score  # , root_mean_squared_error, top_k_accuracy_score

from hybrid_surr import aux_f, plot_funcs, plot_hyb, surr_funcs

"""
For demonstration purposes without large files: search_full=False.
After creating datasets and training, inference requires search_full=True.

After creating datasets, there will be:
- hybrid_surr\num_exp\net_data\<topology>_seir\p_*.csv as seir dataframes for .run_hybrid_model().
- hybrid_surr\num_exp\net_data\<topology>_incidence_100k.csv 
- hybrid_surr\num_exp\net_data\<topology>_beta_100k.csv 
- hybrid_surr\num_exp\hyb_models\*_median_beta.csv

After training, there will be:
- hybrid_surr\num_exp\results\<topology>\<switch>\*.csv as result dataframes for .hybrid_heatmap_r2(), .plot_heatmap_switch()
- hybrid_surr\num_exp\hyb_models\*_lstm_*.keras as trained hybrid models for lstm
- hybrid_surr\num_exp\hyb_models\*_regression_*.joblib as trained hybrid models for regression
- hybrid_surr\num_exp\surr_models\*.pt as trained surrogate models.
"""
search_full = False

# because the surrogate model was saved as a whole and requires the same path
sys.path.append(os.path.abspath("hybrid_surr/num_exp"))


# --------- hybrid ----------


def run_hybrid_model(
    sigma: float = 0.3,
    gamma: float = 0.2,
    switch_I_fraction: float = 0.05,
    n_hybrid_runs: int = 20,
    topology: Literal["ba", "sw"] = "ba",
    beta_pred: list[
        Literal[
            "last value",
            "expanding mean last value",
            "median beta",
            "regression beta",
            "lstm",
        ]
    ] = ["regression beta", "lstm"],
    seir_df_paths: list[str] = [
        "hybrid_surr/num_exp/net-data/ba_seir/p_0.13_0.3_0.2_0.0001_0.39_seed_0.csv",
        "hybrid_surr/num_exp/net-data/ba_seir/p_0.99_0.3_0.2_0.0001_0.63_seed_0.csv",
    ],
    save_results: bool = False,
    res_folder_name: str = "example",
    show_plots: bool = True,
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
) -> tuple[list[Any], list[Any], str, str, str, str]:

    n_unique = len(seir_df_paths) * len(beta_pred)

    if show_plots:
        rows = n_unique // 2 + math.ceil(n_unique % 2)
        fig, ax = plt.subplots(rows, 2, figsize=(10, 3 * rows))
        ax = ax.flatten()
    else:
        ax = np.array([0] * n_unique)

    if topology == "ba":
        suff_m = "ba100k"
        suff = suff_m + f"sI_fullR_{int(switch_I_fraction*100)}"
    if topology == "sw":
        suff_m = "sw100k"
        suff = suff_m + f"sI_fullR_{int(switch_I_fraction*100)}"

    seed_dirs = ""

    types_start_day = ["fraq_people"]
    type_start_day = types_start_day[0]  # for now
    j = 0

    method_label_d = {
        "last value": "last_value",
        "expanding mean last value": "expanding_mean_last_value",
        "median beta": "median_beta",
        "regression beta": "regression_beta",
        "lstm": "lstm_day_E_previous_I",
    }

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if len(alphabet) < n_unique:
        alphabet += "X" * (n_unique - len(alphabet))
    labs = ["(" + alphabet[index] + ")" for index in range(n_unique)][::-1]
    r2s = []
    switches = []
    m_folder = f"{folder_main}/num_exp/"
    models_folder = f"{folder_main}/num_exp/hyb_models/"

    ax = ax.reshape(-1, len(beta_pred))

    for idx_method, method in enumerate(beta_pred):
        if "median" in method:
            model_path = models_folder + f"{suff_m}_median_beta.csv"
        elif "regression beta" in method:
            model_path = models_folder + f"{suff_m}_regression_bt.joblib"
        elif "lstm" in method:
            model_path = models_folder + f"{suff_m}_lstm_4_001_s10"
        else:
            model_path = ""
        try:
            (
                all_rmse_I,
                all_rmse_Inc,
                all_rmse_Beta,
                all_r2,
                all_r2_Inc,
                all_r2_full,
                all_r2_Inc_full,
                all_peak,
                execution_time,
                start_days,
            ) = plot_hyb.main_f(
                I_prediction_method="seir",
                count_stoch_line=n_hybrid_runs,
                beta_prediction_method=method,
                type_start_day=types_start_day[0],
                seed_numbers=seir_df_paths,
                show_fig_flag=show_plots,
                seed_dirs=seed_dirs,
                sigma=sigma,
                gamma=gamma,
                ax=ax[:, idx_method],
                model_path=model_path,
                perc_switch=switch_I_fraction,
                is_filename=False,
                on_incidence=True,
                switch_on_incidence=False,
                topology=topology,
                res_folder_name=res_folder_name,
            )
            r2s.append(all_r2_Inc)
            switches.append(start_days)

            if save_results:
                # creating a dataframe for peaks
                b_a = pd.DataFrame(
                    [
                        [i.split("/")[-1].split("_")[1], i.split("/")[-1].split("_")[5]]
                        for i in seir_df_paths
                    ],
                    dtype=float,
                    columns=["beta", "alpha"],
                )
                all_peak = pd.DataFrame(
                    all_peak,
                    columns=[
                        "actual_peak_I",
                        "predicted_peak_I",
                        "actual_peak_Inc",
                        "predicted_peak_inc",
                        "actual_peak_day",
                        "predicted_peak_day",
                        "actual_peak_day_Inc",
                        "predicted_peak_day_inc",
                    ],
                )
                # creating a dataframe for peaks RMSE, predicted time, start day
                rmse_df = pd.DataFrame(
                    {
                        "rmse_I": all_rmse_I,
                        "rmse_Inc": all_rmse_Inc,
                        "rmse_Beta": all_rmse_Beta,
                        "r2": all_r2,
                        "r2_Inc": all_r2_Inc,
                        "r2_full": all_r2_full,
                        "r2_Inc_full": all_r2_Inc_full,
                        "time_predict": execution_time,
                        f"{type_start_day}": start_days,
                    }
                )

                # merging dataframes
                results = pd.concat([b_a, rmse_df, all_peak], axis=1)
                # seed_numbers.split('/')[0]

        except FileNotFoundError as e:
            raise FileNotFoundError(f"The SEIR file was not found; {e}")
            #pass

        if save_results:
            folder_name = res_folder_name
            path = f"{m_folder}/results/{folder_name}/{type_start_day}/"
            if not os.path.exists(path):
                os.makedirs(path)
            new_label = method_label_d.get(method, "q")
            results.to_csv(f"{path}/{new_label}_results_{suff}.csv", index=False)

    ax = ax.flatten()
    if show_plots:
        for j in range(n_unique):
            ax[j].text(-0.1, 1.1, labs.pop(), transform=ax[j].transAxes, size=15)

    answer_plots, answer_results = (
        "",
        "",
    )
    fig_path, res_path = "", ""
    if show_plots:
        fig_path = f"{folder_imgs}/run_hybrid_model.png"
        plt.savefig(fig_path, bbox_inches="tight")
        answer_plots = ", saved the plots for the hybrid model"
    if save_results:
        answer_results = ", saved the predicted trajectories"
        res_path = f"{folder_main}/num_exp/" + "/results/" + res_folder_name

    r2s = np.array(r2s).flatten().tolist()
    switches = np.array(switches).flatten().tolist()

    return r2s, switches, fig_path, res_path, answer_plots, answer_results


def hybrid_heatmap_r2(
    topology: Literal["ba", "sw"] = "ba",
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
    only_df: bool = False,
) -> str:
    switch_perc = 5
    metric = "r2"
    switch = "fraq_people"
    fin_inc = aux_f.df_metrics(
        folder_name=f"{folder_main}/num_exp/",
        top_name=f"new_{topology}_100000",
        test_suff=f"{topology}_",
        switch=switch,
        with_inc=True,
        trim=False,
        suff=f"_{topology}100k_sI_fullR_{switch_perc}",
    )
    if not only_df:
        aux_f.metric_hmaps(
            folder_name=f"{folder_main}/num_exp/",
            fin=fin_inc,
            met=f"{metric}_Inc",
            suff=f"_{topology}_sI_{switch_perc}",
            exclude=["Cumulative Average", "Exponential Decay"],
            save=False,
        )

        fig_path = f"{folder_imgs}/hybrid_heatmap_r2.png"
        plt.savefig(fig_path, bbox_inches="tight")

        return fig_path

    else:
        return fin_inc


# --------- surrogate ----------


def run_surrogate_point(
    topology: Literal["ba", "sw"] = "ba",
    alphas: list[float] = [0.44, 0.59, 0.71, 0.75],
    betas: list[float] = [0.35, 0.4, 0.41, 0.37],
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
) -> tuple[list[Any], str]:

    type_df = "point"
    ae = torch.load(
        f"{folder_main}/num_exp/surr_models/autoencoder_{topology}_100k_n.pt",
        weights_only=False,
    )
    df = pd.read_csv(
        folder_main + "/num_exp/net_data/" + f"/{topology}_{type_df}_dataset.csv",
        index_col=0,
    )
    df[["beta", "alpha"]] = df[["beta", "alpha"]].round(2)

    fontsize = 12
    rows = len(alphas) // 2 + math.ceil(len(alphas) % 2)

    fig, ax = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    ax = ax.flatten()
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if len(alphabet) < len(alphas):
        alphabet += "X" * (len(alphas) - len(alphabet))
    labels = ["(" + alphabet[index] + ")" for index in range(len(alphas))]
    j = 0
    cut = 100

    r2s = []
    for ax_i, beta, alpha in zip(ax, betas, alphas):
        surrogate_sim = surr_funcs.predict(ae, [beta, alpha]).numpy()
        real_inc = (
            df[(df.beta == beta) & (df.alpha == alpha)]
            .iloc[:, 5:]
            .values.flatten()
            .tolist()
        )
        r2 = r2_score(real_inc, surrogate_sim)
        r2s.append(r2)

        ax_i.plot(real_inc[:cut], label="Network model", marker="o", color="OrangeRed")
        ax_i.plot(
            surrogate_sim[:cut],
            lw=3,
            color="RoyalBlue",
            label=f"Surrogate model\n$R^2=${r2:.3f}",
        )

        ax_i.set_xlabel("Time, days", fontsize=1.2 * fontsize)
        ax_i.set_ylabel("Incidence, cases", fontsize=1.2 * fontsize)
        ax_i.set_ylim(0, 3000)
        ax_i.set_xlim(-5, 100)
        ax_i.tick_params(axis="both", which="major", labelsize=fontsize)
        ax_i.legend(fontsize=1.2 * fontsize)
        ax_i.grid()

        # Add subplot label outside the top-left corner
        ax_i.text(-0.1, 1.1, labels[j], transform=ax_i.transAxes, size=fontsize * 1.5)
        j += 1

    plt.tight_layout()
    fig_path = f"{folder_imgs}/surrogate_point.png"
    plt.savefig(fig_path, bbox_inches="tight")

    r2s = np.array(r2s).flatten().tolist()

    return r2s, fig_path


def run_surrogate_interval(
    topology: Literal["ba", "sw"] = "ba",
    alphas: list[float] = [0.32, 0.59, 0.71, 0.6],
    betas: list[float] = [0.83, 0.4, 0.41, 0.66],
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
) -> tuple[list[Any], str]:

    type_df = "interval"
    ae = torch.load(
        folder_main + f"/num_exp/surr_models/autoencoder_interval_{topology}_100k_n.pt",
        weights_only=False,
    )
    if search_full:
        X_train, y_train, X_test, y_test, tmax, mtest, df_stoch_ts = (
            surr_funcs.get_splits_df(
                folder=folder_main + "calibr/",
                folder_all=folder_main + "num_exp/",
                type_df=type_df,
                network_type=topology,
                with_orig_X=True,
                search_full=search_full,
            )
        )
    else:
        df_stoch_ts = pd.read_csv(
            folder_main + "/num_exp/net_data/" + f"/{topology}_4id_10samples.csv"
        )
    df_stoch_ts[["beta", "alpha"]] = df_stoch_ts[["beta", "alpha"]].round(2)

    # type_df = 'point'
    df_mean_ts = pd.read_csv(
        folder_main + "/num_exp/net_data/" + f"/{topology}_{type_df}_dataset.csv",
        index_col=0,
    )
    df_mean_ts[["beta", "alpha"]] = df_mean_ts[["beta", "alpha"]].round(2)

    fontsize = 11
    rows = len(alphas) // 2 + math.ceil(len(alphas) % 2)
    fig, ax = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    ax = ax.flatten()
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if len(alphabet) < len(alphas):
        alphabet += "X" * (len(alphas) - len(alphabet))
    labels = ["(" + alphabet[index] + ")" for index in range(len(alphas))]

    if topology == "ba":
        tmax = 150
        cut = 100
    elif topology == "sw":
        tmax = 350
        cut = 350
    mean_index = range(tmax)
    low_index = range(tmax, 2 * tmax)
    high_index = range(2 * tmax, 3 * tmax)
    j = 0

    r2s = []
    for ax_i, beta, alpha in zip(ax, betas, alphas):
        surrogate_sim = surr_funcs.predict(ae, [beta, alpha]).numpy()
        real_inc = (
            df_mean_ts[(df_mean_ts.beta == beta) & (df_mean_ts.alpha == alpha)]
            .iloc[:, 5:]
            .values.flatten()
        )  # .tolist()
        r2 = r2_score(real_inc, surrogate_sim)
        r2s.append(r2)

        part = df_stoch_ts[
            (df_stoch_ts.beta == beta) & (df_stoch_ts.alpha == alpha)
        ].iloc[:, 5:-1]
        part.columns = part.columns.astype(int)
        ax_i.plot(
            part.T,
            color="OrangeRed",
            ls=":",
            alpha=0.5,
            label=["Network model, trajectory"] + [""] * 9,
        )
        ax_i.plot(
            real_inc[mean_index][:cut],
            label="Network model, mean",
            marker="",
            color="OrangeRed",
        )
        ax_i.fill_between(
            np.linspace(0, tmax, tmax)[:cut],
            real_inc[low_index][:cut],
            real_inc[high_index][:cut],
            alpha=0.3,
            color="OrangeRed",
            label="Network model interval",
        )

        # ax_i.plot(surrogate_sim[:cut], lw=3, color='RoyalBlue',
        #                  label=f'Surrogate model\n$R^2=${r2:.3f}')
        ax_i.plot(
            surrogate_sim[mean_index][:cut],
            lw=2,
            color="RoyalBlue",
            label="Surrogate model, mean",
        )
        ax_i.fill_between(
            np.linspace(0, tmax, tmax)[:cut],
            surrogate_sim[low_index][:cut],
            surrogate_sim[high_index][:cut],
            alpha=0.3,
            color="RoyalBlue",
            label="Surrogate model interval",
        )

        ax_i.set_xlabel("Time, days", fontsize=1.2 * fontsize)
        ax_i.set_ylabel("Incidence, cases", fontsize=1.2 * fontsize)
        ax_i.set_ylim(0, 2800)
        ax_i.set_xlim(-5, cut)
        ax_i.tick_params(axis="both", which="major", labelsize=fontsize)
        ax_i.legend(fontsize=fontsize)
        ax_i.grid()

        # Add subplot label outside the top-left corner
        ax_i.text(-0.1, 1.1, labels[j], transform=ax_i.transAxes, size=fontsize * 1.5)
        j += 1
    plt.tight_layout()

    fig_path = f"{folder_imgs}/surrogate_interval.png"
    plt.savefig(fig_path, bbox_inches="tight")
    r2s = np.array(r2s).flatten().tolist()

    return r2s, fig_path


def surrogate_heatmap_r2(
    topology: Literal["ba", "sw"] = "ba",
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
    only_df: bool = False,
) -> str:

    type_df = "point"
    ae = torch.load(
        f"{folder_main}/num_exp/surr_models/autoencoder_{topology}_100k_n.pt",
        weights_only=False,
    )
    X_train, y_train, X_test, y_test, tmax = surr_funcs.get_splits_df(
        folder=folder_main + "num_exp/net_data/", type_df=type_df, network_type=topology
    )
    dd = surr_funcs.df_for_heatmap(ae, type_df, X_train, y_train, X_test, y_test, tmax)

    type_df = "interval"
    ae = torch.load(
        f"{folder_main}/num_exp/surr_models/autoencoder_interval_{topology}_100k_n.pt",
        weights_only=False,
    )
    X_train, y_train, X_test, y_test, tmax = surr_funcs.get_splits_df(
        folder=folder_main + "num_exp/net_data/", type_df=type_df, network_type=topology
    )
    dd2_mean, dd2_min, dd2_high = surr_funcs.df_for_heatmap(
        ae, type_df, X_train, y_train, X_test, y_test, tmax
    )

    if not only_df:
        fontsize = 15
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        ax = axes.flatten()

        n = ["(a)", "(b)", "(c)", "(d)"][::-1]

        cmap = surr_funcs.nonlinear_cmap()
        for i, heat_df, title in zip(
            range(4),
            [dd, dd2_mean, dd2_min, dd2_high],
            [
                "Point estimation",
                "Interval estimation (mean)",
                "Interval estimation (lower bound)",
                "Interval estimation (upper bound)",
            ],
        ):

            ax_i = sns.heatmap(
                heat_df.sort_index(level=1, ascending=False),
                cmap=cmap,
                ax=ax[i],  # norm=norm,
                cbar_kws={
                    "extendfrac": 0.1,
                    # "ticks":ticks, "boundaries":boundaries
                },
                vmin=0,
                vmax=1,
                xticklabels=10,
                yticklabels=10,
                linewidths=0.0,
                rasterized=True,
            )
            ax_i.set_title(title, fontsize=1.2 * fontsize)
            ax_i.text(-0.1, 1.1, n.pop(), transform=ax_i.transAxes, size=1.5 * fontsize)
            ax_i.collections[0].cmap.set_bad("0.7")
            ax_i.set_xlabel(r"$\beta_n$", fontsize=1.2 * fontsize)
            ax_i.set_ylabel(r"$\alpha$", fontsize=1.2 * fontsize)
            ax_i.tick_params(axis="both", which="major", labelsize=fontsize)
            cbar = ax_i.collections[0].colorbar
            cbar.set_label(r"$R^2$", rotation=0, size=fontsize)

        for i in [-1, -2, -3, -4]:
            ax_i.figure.axes[i].tick_params(labelsize=fontsize)

        # ax_1.figure.axes[-1].set_ylabel(r'$R^2$', size=fontsize)
        # ax_1.figure.axes[-2].set_ylabel(r'$R^2$', size=fontsize)

        plt.tight_layout()
        fig_path = f"{folder_imgs}/surrogate_heatmap_r2.png"
        plt.savefig(fig_path, bbox_inches="tight")

        return fig_path
    else:
        return dd, dd2_mean, dd2_min, dd2_high


# --------- calibrations/forecasts ----------
def calibrate_model_complete_data(
    model_name: Literal["hybrid", "network", "surrogate"] = "hybrid",
    n_network_runs: int = 1,
    show_surr_nth_line: int = 5,
    sigma: float = 0.3,
    gamma: float = 0.2,
    true_alpha: float = 0.95,
    true_beta: float = 0.1,
    switch_I_fraction: float = 0.05,
    n_nodes: int = 100000,
    topology: Literal["ba", "sw"] = "ba",
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
) -> tuple[Any, Any, Any, str]:

    folder = f"{folder_main}/calibr/"

    if topology == "ba":
        cut = 100
    if topology == "sw":
        cut = 350
    observed_data = pd.read_csv(
        folder + f"observed_incidence_a{true_alpha}_b{true_beta}.csv"
    ).iloc[:cut]
    observed_data.columns = ["incidence"]

    # _________
    with_switch = np.array(True)
    num_runs = [1]

    if model_name == "hybrid":
        model_str = "hyb"
    elif model_name == "network":
        model_str = "net"
    elif model_name == "surrogate":
        num_runs = [0]
        model_str = "surr"

    top = topology
    koeff = 1
    shift = 0
    network_params = [
        with_switch,
        num_runs,
        [switch_I_fraction],
        sigma,
        gamma,
        n_nodes,
        top,
        koeff,
        shift,
    ]

    model_idata = f"{top}_{model_str}_a{true_alpha}_b{true_beta}.nc"

    try:
        # load the calibrated model
        idata = az.from_netcdf(folder + "/models/" + model_idata)
    except FileNotFoundError as e:
        raise NotImplementedError(f"The calibrated model was not found; {e}")

    if model_name == "surrogate":
        idata = idata.rename({"beta": "tau"})

    beta_mode, alpha_mode, best_r2 = plot_funcs.plot_calib(
        observed_data,
        idata,
        true_beta,
        true_alpha,
        network_params,
        n_hyb_runs=n_network_runs,
        nth=show_surr_nth_line,
        model_path=folder_main + "/num_exp/hyb_models/",
    )

    fig_path = f"{folder_imgs}/calibrate_model_complete_data.png"
    plt.savefig(fig_path)

    return alpha_mode, beta_mode, best_r2, fig_path


def calibrate_model_complete_data_3in1(
    n_network_runs: int = 1,
    show_surr_nth_line: int = 5,
    sigma: float = 0.3,
    gamma: float = 0.2,
    true_alpha: float = 0.95,
    true_beta: float = 0.1,
    switch_I_fraction: float = 0.05,
    n_nodes: int = 100000,
    topology: Literal["ba", "sw"] = "ba",
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
) -> tuple[list[Any], list[Any], list[Any], str]:

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    labels = ["(" + alphabet[index] + ")" for index in range(len(alphabet))]

    fig = plt.figure(figsize=(20, 10))
    # adding gridspec
    gs = fig.add_gridspec(
        2, 3, hspace=0.2, width_ratios=[1, 1, 1], height_ratios=[1, 1.25]
    )

    # curves subplots
    gs00 = gs[0, 0].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.0)
    gs01 = gs[0, 1].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.0)
    gs02 = gs[0, 2].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.0)

    # ufo subplots
    gs10 = gs[1, 0].subgridspec(
        2, 2, wspace=0, hspace=0.0, width_ratios=[4, 1], height_ratios=[1, 4]
    )
    gs11 = gs[1, 1].subgridspec(
        2, 2, wspace=0, hspace=0.0, width_ratios=[4, 1], height_ratios=[1, 4]
    )
    gs12 = gs[1, 2].subgridspec(
        2, 2, wspace=0, hspace=0.0, width_ratios=[4, 1], height_ratios=[1, 4]
    )

    folder = f"{folder_main}/calibr/"
    if topology == "ba":
        cut = 100
    if topology == "sw":
        cut = 350

    observed_data = pd.read_csv(
        folder + f"observed_incidence_a{true_alpha}_b{true_beta}.csv"
    ).iloc[:cut]
    observed_data.columns = ["incidence"]

    # _________
    with_switch = np.array(True)
    num_runs = [1]
    top = topology
    koeff = 1
    shift = 0
    network_params = [
        with_switch,
        num_runs,
        [switch_I_fraction],
        sigma,
        gamma,
        n_nodes,
        top,
        koeff,
        shift,
    ]
    idatas = []
    for model_str in ["net", "hyb", "surr"]:
        try:
            # load the calibrated model
            idata = az.from_netcdf(
                folder + "/models/" + f"{top}_{model_str}_a{true_alpha}_b{true_beta}.nc"
            )
            idatas.append(idata)
        except FileNotFoundError as e:
            raise NotImplementedError(f"The calibrated model was not found; {e}")

    # surr model's idata was saved with other arguments
    idatas[-1] = idatas[-1].rename({"beta": "tau"})

    beta_modes, alpha_modes, best_r2s = [], [], []
    for gs_0i, gs_1i, idata_i, idx in zip(
        [gs00, gs01, gs02], [gs10, gs11, gs12], idatas, np.arange(3)
    ):

        if idx == 1:
            with_switch = np.array(True)
            network_params = [
                with_switch,
                num_runs,
                switch_I_fraction,
                sigma,
                gamma,
                n_nodes,
                top,
                koeff,
                shift,
            ]
        elif idx == 2:
            num_runs = [0]
            switch_I_fraction = 1.0
            with_switch = np.array(False)
            network_params = [
                with_switch,
                num_runs,
                [switch_I_fraction],
                sigma,
                gamma,
                n_nodes,
                top,
                koeff,
                shift,
            ]

        ax_curves = fig.add_subplot(gs_0i[0])
        ax_up = fig.add_subplot(gs_1i[0])
        ax_scatter = fig.add_subplot(gs_1i[2], sharex=ax_up)
        ax_right = fig.add_subplot(gs_1i[3], sharey=ax_scatter)

        beta_mode, alpha_mode, best_r2 = plot_funcs.plot_calib(
            observed_data,
            idata_i,
            true_beta,
            true_alpha,
            network_params,
            pred=False,
            nth=show_surr_nth_line,
            n_hyb_runs=n_network_runs,
            ax_curves=[ax_curves],
            ax_kde=[ax_up, ax_scatter, ax_right],
            model_path=folder_main + "/num_exp/hyb_models/",
        )
        fontsize = 14
        ax_curves.annotate(
            labels[idx],
            xy=(0, 0),
            xycoords="axes fraction",
            xytext=(-30, -50),
            textcoords="offset points",
            fontsize=1.5 * fontsize,
            ha="right",
            va="baseline",
        )
        beta_modes.append(beta_mode)
        alpha_modes.append(alpha_mode)
        best_r2s.append(best_r2)

    fig_path = f"{folder_imgs}/calibrate_model_complete_data_3in1.png"
    plt.savefig(fig_path)

    return alpha_modes, beta_modes, best_r2s, fig_path


def calibrate_model_forecast(
    model_name: Literal["hybrid", "surrogate"] = "hybrid",
    show_surr_nth_line: int = 5,
    start_forecasting: Literal["14b", "7b", "7a"] = "14b",
    sigma: float = 0.3,
    gamma: float = 0.2,
    true_alpha: float = 0.95,
    true_beta: float = 0.1,
    switch_I_fraction: float = 0.05,
    n_nodes: int = 100000,
    topology: Literal["ba", "sw"] = "ba",
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
) -> tuple[Any, Any, str]:

    folder = f"{folder_main}/calibr/"
    if topology == "ba":
        cut = 100
    if topology == "sw":
        cut = 350
    observed_data = pd.read_csv(
        folder + f"observed_incidence_a{true_alpha}_b{true_beta}.csv"
    ).iloc[:cut]
    observed_data.columns = ["incidence"]

    with_switch = np.array(False)
    num_runs = [1]
    top = topology
    koeff = 1
    shift = 0
    network_params = [
        with_switch,
        num_runs,
        [switch_I_fraction],
        sigma,
        gamma,
        n_nodes,
        top,
        koeff,
        shift,
    ]

    if model_name == "hybrid":
        model_str = "hyb"
    elif model_name == "surrogate":
        num_runs = [0]
        model_str = "surr"

    model_idata = f"{top}_{model_str}_a{true_alpha}_b{true_beta}_{start_forecasting}.nc"
    try:
        # load the calibrated model
        idata = az.from_netcdf(folder + "/models/" + model_idata)
    except FileNotFoundError as e:
        raise NotImplementedError(f"The calibrated model was not found; {e}")

    beta_mode, alpha_mode, _ = plot_funcs.plot_calib(
        observed_data,
        idata,
        true_beta,
        true_alpha,
        network_params,
        pred=True,
        nth=show_surr_nth_line,
        model_path=folder_main + "/num_exp/hyb_models/",
    )

    fig_path = f"{folder_imgs}/calibrate_model_forecast.png"
    plt.savefig(fig_path)

    return alpha_mode, beta_mode, fig_path


def calibrate_model_forecast_3in1(
    model_name: Literal["hybrid", "surrogate"] = "hybrid",
    show_surr_nth_line: int = 5,
    sigma: float = 0.3,
    gamma: float = 0.2,
    true_beta: float = 0.1,
    true_alpha: float = 0.95,
    switch_I_fraction: float = 0.05,
    n_nodes: int = 100000,
    topology: Literal["ba", "sw"] = "ba",
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
) -> tuple[list[Any], list[Any], str]:

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    labels = ["(" + alphabet[index] + ")" for index in range(len(alphabet))]

    folder = f"{folder_main}/calibr/"
    if topology == "ba":
        cut = 100
    if topology == "sw":
        cut = 350
    observed_data = pd.read_csv(
        folder + f"observed_incidence_a{true_alpha}_b{true_beta}.csv"
    ).iloc[:cut]
    observed_data.columns = ["incidence"]

    with_switch = np.array(False)
    num_runs = [1]
    top = topology
    koeff = 1
    shift = 0
    network_params = [
        with_switch,
        num_runs,
        [switch_I_fraction],
        sigma,
        gamma,
        n_nodes,
        top,
        koeff,
        shift,
    ]

    if model_name == "hybrid":
        model_str = "hyb"
    elif model_name == "surrogate":
        num_runs = [0]
        model_str = "surr"

    fig = plt.figure(figsize=(20, 10))
    # adding gridspec
    gs = fig.add_gridspec(
        2, 3, hspace=0.2, width_ratios=[1, 1, 1], height_ratios=[1, 1.25]
    )

    # curves subplots
    gs00 = gs[0, 0].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.0)
    gs01 = gs[0, 1].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.0)
    gs02 = gs[0, 2].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.0)

    # ufo subplots
    gs10 = gs[1, 0].subgridspec(
        2, 2, wspace=0, hspace=0.0, width_ratios=[4, 1], height_ratios=[1, 4]
    )
    gs11 = gs[1, 1].subgridspec(
        2, 2, wspace=0, hspace=0.0, width_ratios=[4, 1], height_ratios=[1, 4]
    )
    gs12 = gs[1, 2].subgridspec(
        2, 2, wspace=0, hspace=0.0, width_ratios=[4, 1], height_ratios=[1, 4]
    )
    idatas = []
    for start_pred in ["14b", "7b", "7a"]:
        try:
            # load the calibrated model
            idata = az.from_netcdf(
                folder
                + "/models/"
                + f"{top}_{model_str}_a{true_alpha}_b{true_beta}_{start_pred}.nc"
            )
            idatas.append(idata)
        except FileNotFoundError as e:
            raise NotImplementedError(f"The calibrated model was not found; {e}")

    beta_modes, alpha_modes = [], []
    for gs_0i, gs_1i, idata_i, idx in zip(
        [gs00, gs01, gs02], [gs10, gs11, gs12], idatas, np.arange(3)
    ):

        ax_curves = fig.add_subplot(gs_0i[0])

        ax_up = fig.add_subplot(gs_1i[0])
        ax_scatter = fig.add_subplot(gs_1i[2], sharex=ax_up)
        ax_right = fig.add_subplot(gs_1i[3], sharey=ax_scatter)

        beta_mode, alpha_mode, _ = plot_funcs.plot_calib(
            observed_data,
            idata_i,
            true_beta,
            true_alpha,
            network_params,
            pred=True,
            nth=show_surr_nth_line,
            ax_curves=[ax_curves],
            ax_kde=[ax_up, ax_scatter, ax_right],
            model_path=folder_main + "/num_exp/hyb_models/",
        )
        beta_modes.append(beta_mode)
        alpha_modes.append(alpha_mode)
        fontsize = 14
        ax_curves.annotate(
            labels[idx],
            xy=(0, 0),
            xycoords="axes fraction",
            xytext=(-30, -50),
            textcoords="offset points",
            fontsize=1.5 * fontsize,
            ha="right",
            va="baseline",
        )

    fig_path = f"{folder_imgs}/calibrate_model_forecast_3in1.png"
    plt.savefig(fig_path)

    return alpha_modes, beta_modes, fig_path


# --------- heatmaps/aux.figs ----------


def plot_synth_peaks(
    folder_main: str = "hybrid_surr/", folder_imgs: str = "imgs/", only_df: bool = False
) -> str:

    heat_orig = aux_f.heatmap_orig_peaks(
        topology="ba", folder=f"{folder_main}/num_exp/net_data/"
    )

    heat_orig_sw = aux_f.heatmap_orig_peaks(
        topology="sw", folder=f"{folder_main}/num_exp/net_data/"
    )
    if not only_df:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes.flatten()
        aux_f.peaks_hmaps(
            heat_orig,
            with_inc=True,
            title=", Barabasi-Albert",
            ax=[ax[0], ax[2]],
            n=["(a)", "(c)"],
        )

        aux_f.peaks_hmaps(
            heat_orig_sw,
            with_inc=True,
            title=", small world",
            ax=[ax[1], ax[3]],
            n=["(b)", "(d)"],
        )

        fig_path = f"{folder_imgs}/plot_synth_peaks.png"
        plt.savefig(fig_path, bbox_inches="tight")

        return fig_path

    else:
        return heat_orig, heat_orig_sw


def plot_forecast_peak_errors(
    true_alpha: float = 0.95,
    true_beta: float = 0.1,
    topology: Literal["ba", "sw"] = "ba",
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
    only_df: bool = False,
) -> str:

    start_forecasting = ["14b", "7b", "7a"]
    top = topology
    folder = f"{folder_main}/calibr/"
    if topology == "ba":
        cut = 100
    if topology == "sw":
        cut = 350
    observed_data = pd.read_csv(
        folder + f"observed_incidence_a{true_alpha}_b{true_beta}.csv"
    ).iloc[:cut]
    observed_data.columns = ["incidence"]
    model_strs = ["hyb", "surr"]
    idatas = [
        [
            az.from_netcdf(
                folder
                + "/models/"
                + f"{top}_{model_str}_a{true_alpha}_b{true_beta}_{start_pred}.nc"
            )
            for model_str in model_strs
        ]
        for start_pred in start_forecasting
    ]

    pt_preds, ph_preds = aux_f.create_peak_plot(
        folder_name=folder,
        observed_data=observed_data,
        idatas=idatas,
        with_outliers=True,
        same_lims=True,
        figsize=(3.6 * len(idatas), 4),
        x_lim=(-30, 30),
        y_lim=(0, 1.5),
        alpha_m=0.08,
        alpha_area=0.3,
        save=False,
    )
    if not only_df:
        fig_path = f"{folder_imgs}/plot_forecast_peak_errors.png"
        plt.savefig(fig_path, bbox_inches="tight")

        return fig_path
    else:
        return pt_preds, ph_preds


def plot_heatmap_switch(
    topology: Literal["ba", "sw"] = "ba",
    folder_main: str = "hybrid_surr/",
    folder_imgs: str = "imgs/",
    only_df: bool = False,
) -> str:

    switch_perc = 5
    switch = "fraq_people"
    fin_inc = aux_f.df_metrics(
        folder_name=f"{folder_main}/num_exp/",
        top_name=f"new_{topology}_100000",
        test_suff=f"{topology}_",
        switch=switch,
        with_inc=True,
        trim=False,
        suff=f"_{topology}100k_sI_fullR_{switch_perc}",
    )
    if not only_df:
        aux_f.smth_hmaps(fin_inc, 21)

        fig_path = f"{folder_imgs}/plot_heatmap_switch.png"
        plt.savefig(fig_path, bbox_inches="tight")

        return fig_path
    else:
        return fin_inc
