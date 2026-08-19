from hybrid_surr import service


def test_run_hybrid_model():
    seir_df_paths=\
    ['hybrid_surr/num_exp/net_data/ba_seir/p_0.13_0.3_0.2_0.0001_0.39_seed_0.csv',
    'hybrid_surr/num_exp/net_data/ba_seir/p_0.99_0.3_0.2_0.0001_0.63_seed_0.csv']
    r2s, switches, fig_path, res_path, answer_plots, answer_results = (
            service.run_hybrid_model(sigma=0.3, gamma=0.2, 
               switch_I_fraction=0.05, n_hybrid_runs=20, 
               topology='ba', 
              beta_pred=['regression beta', 'lstm'],
              seir_df_paths=seir_df_paths,
                save_results = False,
              res_folder_name='example',
              show_plots = False
            )
        )
    assert switches == [4, 10, 4, 10]
    for i in r2s:
        assert i>0. # for THESE seir examples 


def test_hybrid_heatmap_r2():
    df_inc = service.hybrid_heatmap_r2(topology='ba',
                                         folder_main= "hybrid_surr/",
                                         folder_imgs = "imgs/",
                                        only_df=True)

    assert df_inc.isna().sum().sum() == 0