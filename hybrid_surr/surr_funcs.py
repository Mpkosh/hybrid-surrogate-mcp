import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score, root_mean_squared_error, top_k_accuracy_score
from sklearn.model_selection import train_test_split

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def predict(model, input):
    custom_data_tensor = torch.tensor(input, dtype=torch.float32)
    custom_data_tensor = custom_data_tensor.to(device)

    with torch.no_grad():
        model.eval()
        pred = model(custom_data_tensor)
    return pred.detach().cpu()


def get_splits_df(
    folder="",
    folder_all="",
    type_df="point",
    network_type="ba",
    with_orig_X=False,
    search_full=False,
):
    df = pd.read_csv(folder + f"/{network_type}_{type_df}_dataset.csv", index_col=0)
    if type_df == "point":
        tmax = df.values.shape[1] - 5  # number of simulation days
        # 5 - number of parameters of the network model
        df["ts"] = df[[str(day_index) for day_index in range(tmax)]].values.tolist()
    else:
        tmax = int((df.values.shape[1] - 5) / 3)
        df["ts"] = df[
            ["incidence_" + str(day_index) for day_index in range(tmax)]
            + ["low_incidence_" + str(day_index) for day_index in range(tmax)]
            + ["high_incidence_" + str(day_index) for day_index in range(tmax)]
        ].values.tolist()

    data = df[["beta", "alpha", "ts"]]
    X = data.drop(columns=["ts"])  # [['file']]
    y = data["ts"]

    X_train, X_test = train_test_split(
        X, test_size=2400, random_state=42, stratify=None
    )
    y_train = y.loc[X_train.index].values
    y_test = y.loc[X_test.index].values

    X_train, X_test = X_train.values, X_test.values
    if not with_orig_X:
        return X_train, y_train, X_test, y_test, tmax
    else:
        if search_full:
            # to get 10 stochastic trajectories
            qw = pd.read_csv(
                folder_all + f"{network_type}_incidence_100k.csv"
            ).reset_index(drop=True)
            qw["group"] = qw.index // 10
            mtr, mtest = train_test_split(
                X, test_size=2400, random_state=42, stratify=None
            )
        else:
            qw = pd.read_csv(folder_all + f"{network_type}_4id_10samples.csv")
            mtest = []
        return X_train, y_train, X_test, y_test, tmax, mtest, qw


def nonlinear_cmap():
    def nonlinear_norm(x):
        # Быстрый рост от 0 до 0.8 (линейный)
        # Плавный переход от 0.8 до 0.95 (квадратный корень)
        # Очень медленный рост от 0.95 до 1 (логарифмический)
        return x**4

    # Get plasma colormap and create nonlinear version
    plasma = plt.cm.RdYlGn
    colors = plasma(np.linspace(0, 1, 256))
    new_colors = colors[(nonlinear_norm(np.linspace(0, 1, 256)) * 255).astype(int)]
    nonlinear_cmap = LinearSegmentedColormap.from_list("nonlinear_plasma", new_colors)

    nonlinear_cmap.set_bad("gray", alpha=0.5)  # train in gray
    return nonlinear_cmap


def df_for_heatmap(ae, type_df, X_train, y_train, X_test, y_test, tmax):
    R2_test = []
    R2_mean = []
    R2_min = []
    R2_high = []

    for index, params in enumerate(X_test):
        prediction = predict(ae, params)
        if type_df == "point":
            R2_test.append(r2_score(prediction, y_test[index]))
        elif type_df == "interval":
            R2_mean.append(r2_score(prediction[:tmax], y_test[index][:tmax]))
            R2_min.append(
                r2_score(prediction[tmax : tmax * 2], y_test[index][tmax : tmax * 2])
            )
            R2_high.append(
                r2_score(
                    prediction[tmax * 2 : tmax * 3], y_test[index][tmax * 2 : tmax * 3]
                )
            )

    # np.mean(R2_test),np.mean(R2_mean),np.mean(R2_min),np.mean(R2_high)
    param_1_vals = np.sort(np.unique(np.concatenate((X_train[:, 0], X_test[:, 0]))))
    param_2_vals = np.sort(np.unique(np.concatenate((X_train[:, 1], X_test[:, 1]))))

    # Create empty grid
    if type_df == "point":
        heatmap2 = np.full((len(param_2_vals), len(param_1_vals)), np.nan)
        for (p1, p2), r2 in zip(X_test, R2_test):
            i = np.where(param_2_vals == p2)[0][0]
            j = np.where(param_1_vals == p1)[0][0]
            heatmap2[i, j] = r2
        dd = pd.DataFrame(heatmap2)
        dd.columns = np.arange(0.1, 1.0, 0.01).round(2)
        dd.index = np.arange(0.2, 1.0, 0.01).round(2)
        dd.sort_index(level=1, ascending=False)

        return dd

    elif type_df == "interval":
        heatmap_mean = np.full((len(param_2_vals), len(param_1_vals)), np.nan)
        for (p1, p2), r2 in zip(X_test, R2_mean):
            i = np.where(param_2_vals == p2)[0][0]
            j = np.where(param_1_vals == p1)[0][0]
            heatmap_mean[i, j] = r2
        dd2_mean = pd.DataFrame(heatmap_mean)
        dd2_mean.columns = np.arange(0.1, 1.0, 0.01).round(2)
        dd2_mean.index = np.arange(0.2, 1.0, 0.01).round(2)
        dd2_mean.sort_index(level=1, ascending=False)

        heatmap_min = np.full((len(param_2_vals), len(param_1_vals)), np.nan)
        for (p1, p2), r2 in zip(X_test, R2_min):
            i = np.where(param_2_vals == p2)[0][0]
            j = np.where(param_1_vals == p1)[0][0]
            heatmap_min[i, j] = r2
        dd2_min = pd.DataFrame(heatmap_min)
        dd2_min.columns = np.arange(0.1, 1.0, 0.01).round(2)
        dd2_min.index = np.arange(0.2, 1.0, 0.01).round(2)
        dd2_min.sort_index(level=1, ascending=False)

        heatmap_high = np.full((len(param_2_vals), len(param_1_vals)), np.nan)
        for (p1, p2), r2 in zip(X_test, R2_high):
            i = np.where(param_2_vals == p2)[0][0]
            j = np.where(param_1_vals == p1)[0][0]
            heatmap_high[i, j] = r2
        dd2_high = pd.DataFrame(heatmap_high)
        dd2_high.columns = np.arange(0.1, 1.0, 0.01).round(2)
        dd2_high.index = np.arange(0.2, 1.0, 0.01).round(2)
        dd2_high.sort_index(level=1, ascending=False)

        return dd2_mean, dd2_min, dd2_high
