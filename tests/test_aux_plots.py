from hybrid_surr import service, aux_f


def test_plot_synth_peaks():
    heat_orig, heat_orig_sw = service.plot_synth_peaks(folder_main= "hybrid_surr/",
                                folder_imgs = "imgs/",
                                only_df=True)
    for df in heat_orig, heat_orig_sw:
        assert df.isna().sum().sum() == 0
        assert all(df.actual_peak_Inc>=0)
        assert all(df.actual_peak_day_Inc>=0)


def test_plot_synth_inc_beta():
    mtest_ba, mtest_ba_beta, mtest_sw, mtest_sw_beta= \
        aux_f.plot_synth_inc_beta(
            folder=f"hybrid_surr/num_exp/net_data/", save_folder="imgs/",
            only_df = True)
    for df in mtest_ba, mtest_ba_beta, mtest_sw, mtest_sw_beta:
        assert df.isna().sum().sum() == 0


def test_plot_forecast_peak_errors():
    pt_preds, ph_preds = service.plot_forecast_peak_errors(true_alpha= 0.95,
                              true_beta = 0.1,
                              topology='ba',folder_main= "hybrid_surr/",
                                folder_imgs = "imgs/",
                                only_df = True)
    for one_s in [*pt_preds, *ph_preds]:
        assert one_s.isna().sum() == 0


def test_plot_heatmap_switch():
    fin_inc = service.plot_heatmap_switch(
            topology='ba', folder_main= "hybrid_surr/",
                                folder_imgs = "imgs/",
                                only_df=True
        )
    assert fin_inc.isna().sum().sum() == 0