import pytest

from hybrid_surr import service


def test_calibrate_model_complete_data_3in1():
    alpha_modes, beta_modes, best_r2s, fig_path = (
        service.calibrate_model_complete_data_3in1(
            n_network_runs=1,
            show_surr_nth_line=5000,
            sigma=0.3,
            gamma=0.2,
            true_alpha=0.95,
            true_beta=0.1,
            switch_I_fraction=0.05,
            n_nodes=100000,
            topology="ba",
            folder_main="hybrid_surr/",
            folder_imgs="imgs/",
        )
    )
    assert alpha_modes == [
        pytest.approx(0.876, 0.01),
        pytest.approx(0.894, 0.01),
        pytest.approx(0.885, 0.01),
    ]
    assert beta_modes == [
        pytest.approx(0.161, 0.01),
        pytest.approx(0.149, 0.01),
        pytest.approx(0.151, 0.01),
    ]
    for i in best_r2s:
        assert i > 0.0  # for THESE  examples


def test_calibrate_model_forecast_3in1_hybrid():
    alpha_modes, beta_modes, fig_path = service.calibrate_model_forecast_3in1(
        model_name="hybrid",
        show_surr_nth_line=5,
        sigma=0.3,
        gamma=0.2,
        true_alpha=0.95,
        true_beta=0.1,
        switch_I_fraction=0.05,
        n_nodes=100000,
        topology="ba",
        folder_main="hybrid_surr/",
        folder_imgs="imgs/",
    )
    assert alpha_modes == [
        pytest.approx(0.701, 0.01),
        pytest.approx(0.742, 0.01),
        pytest.approx(0.955, 0.01),
    ]
    assert beta_modes == [
        pytest.approx(0.331, 0.01),
        pytest.approx(0.312, 0.01),
        pytest.approx(0.099, 0.01),
    ]


def test_calibrate_model_forecast_3in1_surrogate():
    alpha_modes, beta_modes, fig_path = service.calibrate_model_forecast_3in1(
        model_name="surrogate",
        show_surr_nth_line=5000,
        sigma=0.3,
        gamma=0.2,
        true_alpha=0.95,
        true_beta=0.1,
        switch_I_fraction=0.05,
        n_nodes=100000,
        topology="ba",
        folder_main="hybrid_surr/",
        folder_imgs="imgs/",
    )
    assert alpha_modes == [
        pytest.approx(0.736, 0.01),
        pytest.approx(0.731, 0.01),
        pytest.approx(0.935, 0.01),
    ]
    assert beta_modes == [
        pytest.approx(0.277, 0.01),
        pytest.approx(0.315, 0.01),
        pytest.approx(0.112, 0.01),
    ]


"""
# 3in1 already covers it
def test_calibrate_model_complete_data():
    alpha_mode, beta_mode, best_r2, fig_path = (
                service.calibrate_model_complete_data(model_name='network',
                        n_network_runs=1, show_surr_nth_line=500,
                        sigma=0.3, gamma=0.2,
                        true_alpha = 0.95,true_beta = 0.1,
                        switch_I_fraction=0.05, n_nodes=100000,
                        topology='ba',
                            folder_main= "hybrid_surr/",
                            folder_imgs = "imgs/",
                )
            )
    assert alpha_mode == pytest.approx(0.876, 0.01)
    assert beta_mode == pytest.approx(0.161, 0.01)
    assert best_r2 > 0 # in this case

    alpha_mode, beta_mode, best_r2, fig_path = (
                    service.calibrate_model_complete_data(model_name='hybrid',
                            n_network_runs=1, show_surr_nth_line=500,
                            sigma=0.3, gamma=0.2,
                            true_alpha = 0.95,true_beta = 0.1,
                            switch_I_fraction=0.05, n_nodes=100000,
                            topology='ba',
                                folder_main= "hybrid_surr/",
                                folder_imgs = "imgs/",
                    )
                )
    assert alpha_mode == pytest.approx(0.894, 0.01)
    assert beta_mode == pytest.approx(0.149, 0.01)
    assert best_r2 > 0 # in this case

    alpha_mode, beta_mode, best_r2, fig_path = (
                    service.calibrate_model_complete_data(model_name='surrogate',
                            n_network_runs=1, show_surr_nth_line=5000,
                            sigma=0.3, gamma=0.2,
                            true_alpha = 0.95,true_beta = 0.1,
                            switch_I_fraction=0.05, n_nodes=100000,
                            topology='ba',
                                folder_main= "hybrid_surr/",
                                folder_imgs = "imgs/",
                    )
                )
    assert alpha_mode == pytest.approx(0.885, 0.01)
    assert beta_mode == pytest.approx(0.151, 0.01)
    assert best_r2 > 0 # in this case
"""
