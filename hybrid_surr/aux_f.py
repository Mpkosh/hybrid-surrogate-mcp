import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy.spatial import ConvexHull
from sklearn.model_selection import train_test_split


def heatmap_orig_peaks(topology="ba", folder=""):
    df = pd.read_csv(
        f"{folder}/{topology}_incidence_100k_10p.csv",
        # skiprows=lambda x: x > 0 and (x - 1) % 10 != 0
    )
    _, df = train_test_split(df, test_size=2400, random_state=42, stratify=None)

    pt = df.iloc[:, 5:].values.argmax(axis=1) + 1
    ph = df.iloc[:, 5:].values.max(axis=1)

    new_df = df.iloc[:, [0, 4]]
    new_df = new_df.round(2)
    new_df["actual_peak_Inc"] = ph
    new_df["actual_peak_day_Inc"] = pt
    return new_df


def get_synth_inc_beta(topology="ba", folder=""):
    qw = pd.read_csv(
        f"{folder}/{topology}_incidence_100k_10p.csv",
        # skiprows=lambda x: x > 0 and (x - 1) % 10 != 0
    )
    X = qw.iloc[:, 5:]
    _, mtest_inc = train_test_split(X, test_size=2400, random_state=42, stratify=None)
    qw_beta = pd.read_csv(f"{folder}/{topology}_beta_100k_10p.csv")
    X_beta = qw_beta.iloc[:, 5:]
    _, mtest_beta = train_test_split(
        X_beta, test_size=2400, random_state=42, stratify=None
    )
    return mtest_inc, mtest_beta


def comma_format(x, pos):
    if x >= 10000:
        return f"{int(x):,}"
    return f"{int(x)}"


def plot_synth_inc_beta(folder="hybrid_surr/num_exp", save_folder="", only_df=False):
    mtest_ba, mtest_ba_beta = get_synth_inc_beta(topology="ba", folder=folder)
    mtest_sw, mtest_sw_beta = get_synth_inc_beta(topology="sw", folder=folder)

    if not only_df:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
        ax = axes.flatten()

        fig2, axes2 = plt.subplots(1, 2, figsize=(7.5, 2.5))
        ax2 = axes2.flatten()

        for mtest, mtest_beta, i, tmax, label in zip(
            [mtest_ba, mtest_sw],
            [mtest_ba_beta, mtest_sw_beta],
            range(2),
            [100, 350],
            ["Barabasi-Albert", "small world"],
        ):
            l1 = ax[i].plot(
                np.arange(tmax), mtest.iloc[::10, :tmax].T, color="RoyalBlue", alpha=0.1
            )
            l2 = ax2[i].plot(
                np.arange(tmax),
                mtest_beta.iloc[::10, :tmax].T,
                color="gray",
                alpha=0.05,
                marker="",
                ls="-",
            )

            ax[i].grid()
            ax2[i].grid()

            # ax[i].axes.xaxis.set_ticklabels([])
            if "Albert" in label:
                ax2[i].set_ylim(0, 3e-5)
                ax[i].set_xlim(0, tmax)
                ax2[i].set_xlim(0, tmax)
            else:
                ax2[i].set_ylim(0, 1e-4)
                ax[i].set_xlim(0, tmax)
                ax2[i].set_xlim(-10, tmax + 10)

            ax[i].set_title(f"Incidence, {label}")
            ax[i].set_ylabel("Incidence, cases")
            ax[i].set_xlabel("Time, days")
            # ax[i].yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
            ax[i].yaxis.set_major_formatter(FuncFormatter(comma_format))

            ax2[i].set_title(rf"$\beta_c$, {label}")
            ax2[i].set_ylabel(r"$\beta_c$")
            ax2[i].set_xlabel("Time, days")

            # ax2[i].set_yscale('linear')
            # ax2[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax2[i].yaxis.set_major_formatter(formatter)

        fig.tight_layout()
        fig2.tight_layout()

        n = ["(a)", "(b)", "(a)", "(b)"][::-1]
        for ax_i in [ax, ax2]:
            for i in range(2):
                ax_i[i].text(
                    -0.1, 1.1, n.pop(), transform=ax_i[i].transAxes, size=1.5 * 8
                )

        # if save_folder:
        path1 = f"{save_folder}/plot_synth_inc.png"
        fig.savefig(path1, bbox_inches="tight")
        path2 = f"{save_folder}/plot_synth_beta.png"
        fig2.savefig(path2, bbox_inches="tight")

        return path1, path2
    else:
        return mtest_ba, mtest_ba_beta, mtest_sw, mtest_sw_beta


def get_mnames():
    clean_mnames = [
        ["Last value", "Cumulative Average", "Median", "Exponential Decay"],
        ["Regression", "LSTM"],
    ]
    methods = [
        ["last_value", "expanding_mean_last_value", "median_beta", "expdecay"],
        ["regression_beta", "lstm_day_E_previous_I"],
    ]
    return clean_mnames, methods


def create_peak_plot(
    folder_name="",
    observed_data="",
    idatas="",
    with_outliers=True,
    same_lims=False,
    figsize=(8, 4),
    x_lim=(-130, 20),
    y_lim=(0.6, 2.3),
    alpha_m=0.4,
    alpha_area=0.35,
    save=False,
):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    labels = ["(" + alphabet[index] + ")" for index in range(len(idatas))]

    fig, axes = plt.subplots(1, len(idatas), figsize=figsize)
    axes = np.array([axes]).flatten()

    size_m = 120

    ph_actual = observed_data.incidence.max()
    pt_actual = observed_data.incidence.argmax()

    ymin, ymax, xmin, xmax = 100, -100, 100, -100
    pt_preds, ph_preds = [], []
    for i, idata_ms, sub_labels in zip(
        np.arange(len(idatas)),
        idatas,
        [["Hybrid", "Surrogate"] for i in range(len(idatas))],
    ):
        pt_pred, ph_pred = plot_peaks_ax(
            axes[i],
            idata_ms,
            sub_labels,
            folder_name,
            ph_actual,
            pt_actual,
            x_lim,
            y_lim,
            size_m,
            alpha_m,
            alpha_area,
            with_outliers,
        )
        pt_preds.append(pt_pred)
        ph_preds.append(ph_pred)
        if same_lims:
            ymin = np.min([ymin, axes[i].get_ylim()[0]])
            ymax = np.max([ymax, axes[i].get_ylim()[1]])
            xmin = np.min([xmin, axes[i].get_xlim()[0]])
            xmax = np.max([xmax, axes[i].get_xlim()[1]])

        fontsize = 9
        axes[i].annotate(
            labels[i],
            xy=(0.12, 1.35),
            xycoords="axes fraction",
            xytext=(-30, -50),
            textcoords="offset points",
            fontsize=1.5 * fontsize,
            ha="right",
            va="baseline",
        )

    if same_lims:
        # pad = 0.2
        for i in np.arange(len(idatas)):
            # if x_min < 0:
            # axes[i].set_xlim(xmin*(1-pad), xmax*(1+pad))
            # axes[i].set_ylim(ymin*(1-pad), ymax*(1+pad))
            axes[i].set_xlim(xmin, xmax)
            axes[i].set_ylim(ymin, ymax)

    axes[0].set_ylabel("Peak incidence ratio")
    plt.tight_layout()

    if save:
        plt.savefig(f"{folder_name}/peaks_area.pdf", format="pdf", bbox_inches="tight")

    return pt_preds, ph_preds


def find_outliers(vals):
    iqr = np.quantile(vals, 0.75) - np.quantile(vals, 0.25)
    l_out = np.quantile(vals, 0.25) - 1.5 * iqr
    h_out = np.quantile(vals, 0.75) + 1.5 * iqr

    if l_out != h_out:
        vals_idx = vals[(l_out < vals) & (vals < h_out)].index
    else:
        vals_idx = vals.index

    return vals_idx, l_out, h_out


def plot_peaks_ax(
    ax,
    idata_ms,
    sub_labels,
    folder_name,
    ph_actual,
    pt_actual,
    x_lim=(-130, 20),
    y_lim=(0.6, 2.3),
    size_m=120,
    alpha_m=0.4,
    alpha_area=0.35,
    with_outliers=True,
):
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1)

    cmap = mpl.colormaps["Set2"]
    colors_l = cmap(np.linspace(0, 1, 8))
    colors = list(colors_l)[1 : 1 + len(idata_ms)]

    for label, idata_m in zip(sub_labels, idata_ms):
        try:
            p_pred = idata_m.predictions.sim.values
            pt_pred = pd.Series(p_pred.argmax(axis=2).flatten() - pt_actual)
            ph_pred = pd.Series(p_pred.max(axis=2).flatten() / ph_actual)

            if not with_outliers:
                ph_idx, l_ph, h_ph = find_outliers(ph_pred)
                pt_idx, l_pt, h_pt = find_outliers(pt_pred)
                clean = list(set(ph_idx).intersection(pt_idx))
                ph_pred = ph_pred.loc[clean]
                pt_pred = pt_pred.loc[clean]

            hull = ConvexHull(pd.concat([pt_pred, ph_pred], axis=1))
            col = colors.pop()
            ax.scatter(
                pt_pred,
                ph_pred,
                marker=".",
                s=size_m,
                alpha=alpha_m,
                label=label,
                zorder=10,
                color=col,
            )

            ax.fill(
                pt_pred.iloc[hull.vertices],
                ph_pred.iloc[hull.vertices],
                alpha=alpha_area,
                color=col,
            )

        except FileNotFoundError:
            pass
            # print(f'---- No data for {name} ----')

    ax.grid()
    ax.set_xlabel("Peak time difference")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    leg = ax.legend(prop={"size": 13}, loc="best")
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    leg.set_zorder(20)
    return pt_pred, ph_pred


def df_metrics(
    folder_name,
    top_name,
    test_suff="",
    switch="",
    with_inc=False,
    trim=False,
    suff="",
    methods=[],
    test_file=True,
):
    if not methods:
        methods = [
            "last_value",
            "median_beta",
            "regression_beta",
            "lstm_day_E_previous_I",
        ]

    fin_m = ["r2", "rmse_I", "rmse_Beta", "pt_err", "ph_err", "time_predict"]
    if with_inc:
        fin_m = [
            "r2",
            "r2_Inc",
            "r2_full",
            "r2_Inc_full",
            "rmse_I",
            "rmse_Inc",
            "rmse_Beta",
            "pt_err",
            "ph_err",
            "pt_err_Inc",
            "ph_err_Inc",
            "time_predict",
        ]

    for label in methods:

        try:
            df = pd.read_csv(
                f"{folder_name}/results/{top_name}/{switch}/"
                + f"{label}_results{suff}.csv"
            )

            if label == methods[0]:
                fin = df[["beta", "alpha"]].astype(float).round(2)
            if trim:
                df = df[df[switch] != 0]
            df["pt_err"] = df["predicted_peak_day"] - df["actual_peak_day"]
            df["ph_err"] = df["predicted_peak_I"] / df["actual_peak_I"]
            if with_inc:
                df["pt_err_Inc"] = (
                    df["predicted_peak_day_inc"] - df["actual_peak_day_Inc"]
                )
                df["ph_err_Inc"] = df["predicted_peak_inc"] / df["actual_peak_Inc"]

            for met in fin_m:
                fin[f"{met}.{label}"] = df[met]

        except FileNotFoundError:

            pass

    fin["switch"] = df[switch]
    fin["days_before_peak"] = df["actual_peak_day"] - df[switch]
    fin["actual_peak_I"] = df["actual_peak_I"]
    fin["actual_peak_day"] = df["actual_peak_day"]

    if with_inc:
        fin["actual_peak_Inc"] = df["actual_peak_Inc"]
        fin["actual_peak_day_Inc"] = df["actual_peak_day_Inc"]

    return fin


def flatten(xss):
    return [x for xs in xss for x in xs]


# Andrew's
def nonlinear_norm(x):
    # Быстрый рост от 0 до 0.8 (линейный)
    # Плавный переход от 0.8 до 0.95 (квадратный корень)
    # Очень медленный рост от 0.95 до 1 (логарифмический)
    return x**4


def metric_hmaps(
    folder_name, fin, met, suff="", exclude=[], figsize=(15, 10), size=15, save=False
):
    clean_mnames, methods = get_mnames()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(5, 3)
    n = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"][::-1]

    nice_label = ""
    if "r2" in met:
        nice_label = r"$R^2$"

    cm = plt.cm.RdYlGn
    colors = cm(np.linspace(0, 1, 256))
    new_colors = colors[(nonlinear_norm(np.linspace(0, 1, 256)) * 255).astype(int)]
    nonlinear_cmap = LinearSegmentedColormap.from_list("nonlinear_plasma", new_colors)

    rows = [0, 0, 2, 2, 1][::-1]
    # rows = [0,0,0,2,2,2][::-1]
    cols = [0, 1, 0, 1, 2][::-1]
    # cols = [0,1,2,0,1,2][::-1]

    for method, label in zip(flatten(methods), flatten(clean_mnames)):
        if label not in exclude:
            try:
                data = fin.pivot(
                    columns="beta", index="alpha", values=f"{met}.{method}"
                )
                r = rows.pop()
                c = cols.pop()
                ax_i = plt.subplot(gs[r : r + 2, c])
                sns.heatmap(
                    data.sort_index(level=1, ascending=False),
                    vmin=0,
                    vmax=1,
                    cmap=nonlinear_cmap,
                    ax=ax_i,
                    yticklabels=int(size * 2 / 3),
                    xticklabels=int(size * 2 / 3),
                    linewidths=0.0,
                    rasterized=True,
                    # cbar_kws={'label': nice_label}
                )
                ax_i.collections[0].cmap.set_bad("0.7")
                ax_i.set_xlabel(r"$\beta_n$")
                ax_i.set_ylabel(r"$\alpha$")

                ax_i.set_title(label, size=size)
                ax_i.text(-0.1, 1.1, n.pop(), transform=ax_i.transAxes, size=size)
                cbar = ax_i.collections[0].colorbar
                cbar.set_label(nice_label, rotation=0)

            except KeyError:
                pass

    plt.tight_layout(w_pad=-0.5, h_pad=-0.5)

    if save:
        plt.savefig(f"results/hmap{suff}.pdf", format="pdf", bbox_inches="tight")


def peaks_hmaps(fin, ax=[], n=["a)", "b)"], with_inc=False, title=""):
    fontsize = 14
    if len(ax) == 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes.flatten()

    cmap = mpl.cm.RdYlGn
    n = n[::-1]

    if with_inc:
        suff = "_Inc"
    else:
        suff = ""

    # bounds = [0, 4, 20, 200]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    data = fin.pivot(columns="beta", index="alpha", values=f"actual_peak_day{suff}")
    ax_1 = sns.heatmap(
        data.sort_index(level=1, ascending=False),
        cmap=cmap,
        ax=ax[0],  # norm=norm,
        cbar_kws={"extendfrac": 0.1},
        # vmax=70,
        xticklabels=10,
        yticklabels=10,
        linewidths=0.0,
        rasterized=True,
    )
    ax_1.set_title("Peak time" + title, fontsize=1.2 * fontsize)
    """
    colorbar = ax_1.collections[0].colorbar
    tick_locs = np.linspace(bounds[0], bounds[-1], 
                            2 * len(bounds) + 1)[1::2]
    colorbar.set_ticks(np.mean([bounds[1:], bounds[:-1]], 0))
    colorbar.set_ticklabels([f'[1, {bounds[1]})', 
                             f'[{bounds[1]}, {bounds[2]-1})',
                             f'[{bounds[2]}, 150)'])
    
    """

    data = (
        fin.pivot(columns="beta", index="alpha", values=f"actual_peak_I{suff[2:]}")
        / 100000
    )
    ax_2 = sns.heatmap(
        data.sort_index(level=1, ascending=False),
        cmap=cmap,
        ax=ax[1],  # norm=norm,
        cbar_kws={"extendfrac": 0.1},
        xticklabels=10,
        yticklabels=10,
        linewidths=0.0,
        rasterized=True,
    )
    ax_2.set_title("Peak incidence" + title, fontsize=1.2 * fontsize)

    for ax_i in [ax_1, ax_2]:
        ax_i.text(-0.1, 1.1, n.pop(), transform=ax_i.transAxes, size=1.5 * fontsize)
        ax_i.collections[0].cmap.set_bad("0.7")
        ax_i.set_xlabel(r"$\beta_n$", fontsize=1.2 * fontsize)
        ax_i.set_ylabel(r"$\alpha$", fontsize=1.2 * fontsize)
        ax_i.tick_params(axis="both", which="major", labelsize=fontsize)
    for i in [-1, -2]:
        ax_1.figure.axes[i].tick_params(labelsize=fontsize)

    ax_1.figure.axes[-1].set_ylabel("Incidence, fraction of population", size=fontsize)
    ax_1.figure.axes[-2].set_ylabel("Day", size=fontsize)

    plt.tight_layout()
    # plt.savefig(f'results/actual.pdf', format='pdf', bbox_inches='tight')


def smth_hmaps(fin, vmax=14):
    fontsize = 14
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.flatten()

    cmap = mpl.cm.RdYlGn
    n = ["(a)", "(b)"][::-1]

    # ticks=np.arange(1,22)
    # boundaries = np.arange(1-.5, 21+1.5 )

    data = fin.pivot(columns="beta", index="alpha", values=f"days_before_peak")
    ax_1 = sns.heatmap(
        data.sort_index(level=1, ascending=False),
        cmap=cmap,
        ax=ax[0],  # norm=norm,
        cbar_kws={
            "extendfrac": 0.1,
            # "ticks":ticks, "boundaries":boundaries
        },
        vmax=vmax,
        xticklabels=10,
        yticklabels=10,
        linewidths=0.0,
        rasterized=True,
    )
    ax_1.set_title("Days from switch to peak", fontsize=1.2 * fontsize)

    colorbar = ax_1.collections[0].colorbar
    """
    tick_locs = np.linspace(bounds[0], bounds[-1], 
                            2 * len(bounds) + 1)[1::2]
    colorbar.set_ticks(np.mean([bounds[1:], bounds[:-1]], 0))
    colorbar.set_ticklabels([f'[1, {bounds[1]})', 
                             f'[{bounds[1]}, {bounds[2]-1})',
                             f'[{bounds[2]}, 150)'])
    """
    data = fin.pivot(columns="beta", index="alpha", values=f"switch")
    ax_2 = sns.heatmap(
        data.sort_index(level=1, ascending=False),
        cmap=cmap,
        ax=ax[1],  # norm=norm,
        cbar_kws={"extendfrac": 0.1},
        vmax=vmax,
        xticklabels=10,
        yticklabels=10,
        linewidths=0.0,
        rasterized=True,
    )
    ax_2.set_title("Day of switch", fontsize=1.2 * fontsize)

    colorbar = ax_2.collections[0].colorbar
    """
    tick_locs = np.linspace(bounds[0], bounds[-1], 
                            2 * len(bounds) + 1)[1::2]
    colorbar.set_ticks(np.mean([bounds[1:], bounds[:-1]], 0))
    
    colorbar.set_ticklabels([r'(0, 1%)', 
                             r'[1%, 5%)', 
                             f'[5%, 10%)', 
                             '[10%, 100%)'])
    """

    for ax_i in [ax_1, ax_2]:
        ax_i.text(-0.1, 1.1, n.pop(), transform=ax_i.transAxes, size=1.5 * fontsize)
        ax_i.collections[0].cmap.set_bad("0.7")
        ax_i.set_xlabel(r"$\beta_n$", fontsize=1.2 * fontsize)
        ax_i.set_ylabel(r"$\alpha$", fontsize=1.2 * fontsize)
        ax_i.tick_params(axis="both", which="major", labelsize=fontsize)
    for i in [-1, -2]:
        ax_1.figure.axes[i].tick_params(labelsize=fontsize)

    ax_1.figure.axes[-1].set_ylabel("Day", size=fontsize)
    ax_1.figure.axes[-2].set_ylabel("Day", size=fontsize)

    plt.tight_layout()
    # plt.savefig(f'results/actual.pdf', format='pdf', bbox_inches='tight')
