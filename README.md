# hybrid-surrogate-mcp

## Tools exposed via MCP

### Hybrid model tools
- `run_hybrid_methods` — run the hybrid model, save results, save plots.
- `hybrid_heatmap_r2` — create several heatmaps with values of coefficient of determination.

### Surrogate model tools
- `surrogate_point` — run the surrogate model with point estimation, save plots.
- `surrogate_interval` — run the surrogate model with interval estimation, save plots.
- `surrogate_heatmap_r2` — create several heatmaps with values of coefficient of determination.

### Calibration / forecasting
- `calibrate_model_complete_data` — calibrate parameters of a chosen model to a target incidence curve and save plots.
- `calibrate_model_complete_data_3in1` — calibrate parameters of three models (network, hybrid, surrogate) to a target incidence curve and save plots.
- `calibrate_model_forecast` — short-time forecasting; calibrate parameters of a chosen model to an incomplete target incidence curve and save plots.
- `calibrate_model_forecast_3in1` — short-time forecasting; calibrate parameters of three models (network, hybrid, surrogate) to an incomplete target incidence curve and save plots.

### Other plotting tools
- `plot_synth_peaks` — create several heatmaps showing the distribution of peak time and peak incidence for synthetic incidence curves.
- `plot_synth_inc_beta` — create plots showing incidence trajectories and beta (infection transmission rate) trajectories.
- `plot_forecast_peak_errors` — create plots showing peak errors for hybrid and surrogate approaches for short-term forecasting.
- `plot_heatmap_switch` — create several heatmaps showing switching behavior for test samples: difference between epidemic peak time and day of switch; distribution of switch days across all runs.

## Files used for creating plots
```
hybrid-surrogate-mcp/
    └── hybrid_surr/
        ├── num_exp/
        │   ├── models/                               # Trained surrogate models 
        │   │   ├── autoencoder_ba_100k_n.pt              # Model with point estimation; used for .run_surrogate_point(), .surrogate_heatmap_r2()
        │   │   └── autoencoder_interval_ba_100k_n.pt     # Model with interval estimation; used for .run_surrogate_interval(), .surrogate_heatmap_r2()
        │   ├── results/                              # Hybrid model results; used for .hybrid_heatmap_r2(), .plot_heatmap_switch()
        │   │   ├── new_ba_100000/                        # Barabasi-Albert results
        │   │   │   └── fraq_people/
        │   │   │       └── ...
        │   │   └── new_sw_100000/                        # Small world results
        │   │       └── fraq_people/
        │   │           └── ...
        │   ├── ba_4id_10samples.csv                  # 10 network runs for 4 alpha/beta pairs; used for .run_surrogate_interval()
        │   ├── ba_beta_100k_10p.csv                  # Network beta trajectories, 10% sample; used for .plot_synth_inc_beta()
        │   ├── ba_incidence_100k_10p.csv             # Network incidence trajectories, 10% sample; used for .plot_synth_inc_beta()
        │   ├── ba_interval_dataset.csv               # Interval network incidence trajectories; used for .run_surrogate_interval(), .surrogate_heatmap_r2()
        │   ├── ba_point_dataset.csv                  # Network incidence trajectories; used for .run_surrogate_point(), .surrogate_heatmap_r2()
        │   ├── ba100k_lstm_4_001_s10.keras           # Trained LSTM model; used for .run_hybrid_model()
        │   ├── ba100k_lstm_4_001_s10.pkl             # Scaler; used for .run_hybrid_model()
        │   ├── ba100k_median_beta.csv                # Median beta of train network runs; used for .run_hybrid_model()
        │   ├── ba100k_regression_bt.joblib           # Trained regression model; used for .run_hybrid_model()
        │   ├── p_0.13_0.3_0.2_0.0001_0.39_seed_0.csv # Example SEIR dataframe; used for .run_hybrid_model()
        │   ├── p_0.99_0.3_0.2_0.0001_0.63_seed_0.csv # Example SEIR dataframe; used for .run_hybrid_model()
        │   ├── sw_beta_100k_10p.csv                  # Network beta trajectories, 10% sample; used for .plot_synth_inc_beta()
        │   └── sw_incidence_100k_10p.csv             # Network beta trajectories, 10% sample; used for .plot_synth_inc_beta()
        └── calibr/
            ├── ba_hyb_a0.95_b0.1_14b.nc              # Calibrated hybrid model, 14 days before peak incidence; used for .calibrate_model_forecast()/...3in1()
            ├── ba_hyb_a0.95_b0.1_7a.nc               # Calibrated hybrid model, 7 days after peak incidence; used for .calibrate_model_forecast()/...3in1()
            ├── ba_hyb_a0.95_b0.1_7b.nc               # Calibrated hybrid model, 7 days before peak incidence; used for .calibrate_model_forecast()/...3in1()
            ├── ba_hyb_a0.95_b0.1.nc                  # Calibrated hybrid model; used for .calibrate_model_complete_data()/...3in1()
            ├── ba_net_a0.95_b0.1.nc                  # Calibrated network model; used for .calibrate_model_complete_data()/...3in1()
            ├── ba_surr_a0.95_b0.1_14b.nc             # Calibrated surrogate model, 14 days before peak incidence; used for .calibrate_model_forecast()/...3in1()
            ├── ba_surr_a0.95_b0.1_7a.nc              # Calibrated surrogate model, 7 days after peak incidence; used for .calibrate_model_forecast()/...3in1()
            ├── ba_surr_a0.95_b0.1_7b.nc              # Calibrated surrogate model, 7 days before peak incidence; used for .calibrate_model_forecast()/...3in1()
            ├── ba_surr_a0.95_b0.1.nc                 # Calibrated surrogate model; used for .calibrate_model_complete_data()/...3in1()
            └── observed_incidence_a0.95_b0.1.csv     # Target incidence curve for calibration
```
