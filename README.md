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
- `calibrate_model_forecast` — short-term forecasting; calibrate parameters of a chosen model to an incomplete target incidence curve and save plots.
- `calibrate_model_forecast_3in1` — short-term forecasting; calibrate parameters of three models (network, hybrid, surrogate) to an incomplete target incidence curve and save plots.

### Other plotting tools
- `plot_synth_peaks` — create several heatmaps showing the distribution of peak time and peak incidence for synthetic incidence curves.
- `plot_synth_inc_beta` — create plots showing incidence trajectories and beta (infection transmission rate) trajectories.
- `plot_forecast_peak_errors` — create plots showing peak errors for hybrid and surrogate approaches for short-term forecasting.
- `plot_heatmap_switch` — create several heatmaps showing switching behavior for test samples: difference between epidemic peak time and day of switch; distribution of switch days across all runs.

## Files used for creating plots
```
hybrid-surrogate-mcp/
├── hybrid_surr/
│   ├── calibr/
│   │   ├── models/                                # Calibrated hybrid/network/surrogate models
│   │   │   └── ...
│   │   └── observed_incidence_a0.95_b0.1.csv      # Target incidence curve for calibration
│   ├── num_exp/
│   │   ├── hyb_models/                            # Trained beta prediction models for hybrid models
│   │   │   └── ...
│   │   ├── net_data/
│   │   │   ├── ba_seir/                               # Example SEIR dataframes
│   │   │   │   └── ...
│   │   │   ├── ba_4id_10samples.csv                   # Example 10 network runs for 4 alpha/beta pairs
│   │   │   ├── ba_beta_100k_10p.csv                   # Network beta trajectories, Barabasi-Albert, 10% sample
│   │   │   ├── ba_incidence_100k_10p.csv              # Network incidence trajectories, Barabasi-Albert, 10% sample
│   │   │   ├── ba_interval_dataset.csv                # Interval network incidence trajectories; for surrogate models
│   │   │   ├── ba_point_dataset.csv                   # Network incidence trajectories; for surrogate models
│   │   │   ├── sw_beta_100k_10p.csv                   # Network beta trajectories, small world, 10% sample
│   │   │   └── sw_incidence_100k_10p.csv              # Network incidence trajectories, small world, 10% sample
│   │   ├── results/                               # Hybrid model results
│   │   │   ├── example/                               # Example results
│   │   │   │   └── fraq_people/
│   │   │   │       └── ...
│   │   │   ├── new_ba_100000/                         # Barabasi-Albert results
│   │   │   │   └── fraq_people/
│   │   │   │       └── ...
│   │   │   └── new_sw_100000/                         # Small world results
│   │   │       └── fraq_people/
│   │   │           └── ...
│   │   ├── surr_models/                           # Trained surrogate models 
│   │   └── ...
│   └── ...
├── imgs/                                  # Example images after tool calling
│   └── ...
└── ...
```
