from hybrid_surr import service


def test_run_surrogate_point():
    r2s, fig_path = service.run_surrogate_point(
        topology="ba",
        alphas=[0.44, 0.59, 0.71, 0.75],
        betas=[0.35, 0.4, 0.41, 0.37],
        folder_main="hybrid_surr/",
        folder_imgs="imgs/",
    )

    for i in r2s:
        assert i > 0.0  # for THESE examples


def test_run_surrogate_interval():
    r2s, fig_path = service.run_surrogate_interval(
        topology="ba",
        alphas=[0.44, 0.59, 0.71, 0.75],
        betas=[0.35, 0.4, 0.41, 0.37],
        folder_main="hybrid_surr/",
        folder_imgs="imgs/",
    )

    for i in r2s:
        assert i > 0.0  # for THESE examples


def test_surrogate_heatmap_r2():
    dd, dd2_mean, dd2_min, dd2_high = service.surrogate_heatmap_r2(
        topology="ba", only_df=True
    )

    for df in dd, dd2_mean, dd2_min, dd2_high:
        assert df.notna().sum().sum() == 2400  # for THESE examples
