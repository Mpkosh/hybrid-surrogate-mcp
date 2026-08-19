#!/usr/bin/env python3
import logging
from typing import Literal

from fastmcp import FastMCP

from hybrid_surr import aux_f, service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Hybrid and Surrogate Server")
folder_main = "hybrid_surr/"
folder_imgs = "imgs/"


# --------- hybrid ----------
@mcp.tool()
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

    r2s, switches, fig_path, res_path, answer_plots, answer_results = (
        service.run_hybrid_model(
            sigma=sigma,
            gamma=gamma,
            switch_I_fraction=switch_I_fraction,
            n_hybrid_runs=n_hybrid_runs,
            topology=topology,
            beta_pred=beta_pred,
            seir_df_paths=seir_df_paths,
            save_results=save_results,
            res_folder_name=res_folder_name,
            show_plots=show_plots,
            folder_main=folder_main,
            folder_imgs=folder_imgs,
        )
    )
    metadata = {
        "R2 for test samples": r2s,
        "Days of switch for test samples": switches,
        "Path to the saved figure": fig_path,
        "Path to the saved results": res_path,
    }
    return {
        "answer": f"Ran the hybrid model{answer_plots}{answer_results}.",
        "metadata": metadata,
    }


@mcp.tool()
def hybrid_heatmap_r2(
    topology: Literal["ba", "sw"] = "ba",
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
    fig_path = service.hybrid_heatmap_r2(
        topology=topology, folder_main=folder_main, folder_imgs=folder_imgs
    )
    metadata = {"Path to the saved figure": fig_path}
    return {
        "answer": "Created the heatmap plot for the hybrid model.",
        "metadata": metadata,
    }


# --------- surrogate ----------
@mcp.tool()
def run_surrogate_point(
    topology: Literal["ba", "sw"] = "ba",
    alphas: list[float] = [0.44, 0.59, 0.71, 0.75],
    betas: list[float] = [0.35, 0.4, 0.41, 0.37],
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

    r2s, fig_path = service.run_surrogate_point(
        topology=topology,
        alphas=alphas,
        betas=betas,
        folder_main=folder_main,
        folder_imgs=folder_imgs,
    )
    metadata = {"R2 for test samples": r2s, "Path to the saved figure": fig_path}
    return {
        "answer": "Created a 2x2 plot for the surrogate model"
        + " with point estimation.",
        "metadata": metadata,
    }


@mcp.tool()
def run_surrogate_interval(
    topology: Literal["ba", "sw"] = "ba",
    alphas: list[float] = [0.32, 0.59, 0.71, 0.6],
    betas: list[float] = [0.83, 0.4, 0.41, 0.66],
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

    r2s, fig_path = service.run_surrogate_interval(
        topology=topology,
        alphas=alphas,
        betas=betas,
        folder_main=folder_main,
        folder_imgs=folder_imgs,
    )

    metadata = {"R2 for test samples": r2s, "Path to the saved figure": fig_path}
    return {
        "answer": "Created the 2x2 plot for the surrogate model "
        + "with interval estimation.",
        "metadata": metadata,
    }


@mcp.tool()
def surrogate_heatmap_r2(
    topology: Literal["ba", "sw"] = "ba",
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

    fig_path = service.surrogate_heatmap_r2(
        topology=topology, folder_main=folder_main, folder_imgs=folder_imgs
    )

    metadata = {
        "Path to the saved figure": fig_path,
    }
    return {
        "answer": "Created the heatmap plot for the surrogate model.",
        "metadata": metadata,
    }


# --------- calibrations/forecasts ----------
@mcp.tool()
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
) -> dict:
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

    alpha_mode, beta_mode, best_r2, fig_path = service.calibrate_model_complete_data(
        model_name=model_name,
        n_network_runs=n_network_runs,
        show_surr_nth_line=show_surr_nth_line,
        sigma=sigma,
        gamma=gamma,
        true_alpha=true_alpha,
        true_beta=true_beta,
        switch_I_fraction=switch_I_fraction,
        n_nodes=n_nodes,
        topology=topology,
        folder_main=folder_main,
        folder_imgs=folder_imgs,
    )

    metadata = {
        "Selected alpha value": alpha_mode,
        "Selected beta value": beta_mode,
        "Best R2": best_r2,
        "Path to the saved figure": fig_path,
    }
    return {
        "answer": f"Calibrated the {model_name} model and created the plot.",
        "metadata": metadata,
    }


@mcp.tool()
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
) -> dict:
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
    alpha_modes, beta_modes, best_r2s, fig_path = (
        service.calibrate_model_complete_data_3in1(
            n_network_runs=n_network_runs,
            show_surr_nth_line=show_surr_nth_line,
            sigma=sigma,
            gamma=gamma,
            true_alpha=true_alpha,
            true_beta=true_beta,
            switch_I_fraction=switch_I_fraction,
            n_nodes=n_nodes,
            topology=topology,
            folder_main=folder_main,
            folder_imgs=folder_imgs,
        )
    )

    metadata = {
        "Selected alpha values": alpha_modes,
        "Selected beta values": beta_modes,
        "Best R2": best_r2s,
        "Path to the saved figure": fig_path,
    }
    return {
        "answer": "Calibrated the network, hybrid and surrogate models "
        + "and created the plot.",
        "metadata": metadata,
    }


@mcp.tool()
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
) -> dict:
    """
    The tool conducts short-time forecasting of disease incidence.
    It calibrates parameters of a chosen model to an incomplete target
    incidence curve simulated by the network model. The calibration employs
    Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC).

    Args:
        model_name (str): The model to use.
            - "hybrid",
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

    alpha_mode, beta_mode, fig_path = service.calibrate_model_forecast(
        model_name=model_name,
        show_surr_nth_line=show_surr_nth_line,
        start_forecasting=start_forecasting,
        sigma=sigma,
        gamma=gamma,
        true_alpha=true_alpha,
        true_beta=true_beta,
        switch_I_fraction=switch_I_fraction,
        n_nodes=n_nodes,
        topology=topology,
        folder_main=folder_main,
        folder_imgs=folder_imgs,
    )

    metadata = {
        "Selected alpha value": alpha_mode,
        "Selected beta value": beta_mode,
        "Path to the saved figure": fig_path,
    }
    return {
        "answer": f"Forecasted with the {model_name} model and created the plot.",
        "metadata": metadata,
    }


@mcp.tool()
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
) -> dict:
    """
    The tool conducts short-time forecasting of disease incidence.
    It calibrates parameters of the chosen model to an incomplete target
    incidence curve simulated by the network model. The calibration employs
    Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC).

    Args:
        model_name (str): The model to use.
            - "hybrid",
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
    alpha_modes, beta_modes, fig_path = service.calibrate_model_forecast_3in1(
        model_name=model_name,
        show_surr_nth_line=show_surr_nth_line,
        sigma=sigma,
        gamma=gamma,
        true_alpha=true_alpha,
        true_beta=true_beta,
        switch_I_fraction=switch_I_fraction,
        n_nodes=n_nodes,
        topology=topology,
        folder_main=folder_main,
        folder_imgs=folder_imgs,
    )

    metadata = {
        "Selected alpha values": alpha_modes,
        "Selected beta values": beta_modes,
        "Path to the saved figure": fig_path,
    }
    return {
        "answer": f"Forecasted with the {model_name} model and created the plot.",
        "metadata": metadata,
    }


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
    fig_path = service.plot_synth_peaks(
        folder_main=folder_main, folder_imgs=folder_imgs
    )
    metadata = {
        "Path to the saved figure": fig_path,
    }
    return {
        "answer": "Created the heatmaps showing the distribution of"
        + " peak time and peak incidence.",
        "metadata": metadata,
    }


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
    fig_path_inc, fig_path_beta = aux_f.plot_synth_inc_beta(
        folder=f"{folder_main}/num_exp/net_data/", save_folder="imgs/"
    )
    metadata = {
        "Paths to the saved figures": [fig_path_inc, fig_path_beta],
    }

    return {
        "answer": "Created the plot showing incidence and beta trajectories.",
        "metadata": metadata,
    }


@mcp.tool()
def plot_forecast_peak_errors(
    true_alpha: float = 0.95,
    true_beta: float = 0.1,
    topology: Literal["ba", "sw"] = "ba",
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

    fig_path = service.plot_forecast_peak_errors(
        true_alpha=true_alpha,
        true_beta=true_beta,
        topology=topology,
        folder_main=folder_main,
        folder_imgs=folder_imgs,
    )

    metadata = {
        "Path to the saved figure": fig_path,
    }
    return {
        "answer": "Created the plot showing peak errors for "
        + "hybrid and surrogate approaches for short-term forecasting.",
        "metadata": metadata,
    }


@mcp.tool()
def plot_heatmap_switch(
    topology: Literal["ba", "sw"] = "ba",
) -> dict:
    """
    The tool creates several heatmaps showing switching behavior for test
    samples: difference between epidemic peak time and day of switch;
    distribution of switch days across all runs.

    Returns:
        dict: A dictionary containing the short result description and metadata
            about the request (path to the saved figure).
    """
    fig_path = service.plot_heatmap_switch(
        topology=topology, folder_main=folder_main, folder_imgs=folder_imgs
    )

    metadata = {
        "Path to the saved figure": fig_path,
    }
    return {
        "answer": "Created the plot showing switching behavior for test samples.",
        "metadata": metadata,
    }


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7331, path="/mcp")
