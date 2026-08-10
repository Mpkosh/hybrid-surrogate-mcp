# hybrid-surrogate-mcp

## Tools exposed via MCP

### Hybrid model tools
- `run_hybrid_methods` — run the hybrid model, save results, save plots.
- `hybrid_2x2` — run the hybrid model and create a 2x2 figure.
- `hybrid_heatmap_r2` - create several heatmaps with values of coefficient of determination.

### Surrogate model tools
- `surrogate_point_2x2` — run the surrogate model with point estimation and create a 2x2 figure.
- `surrogate_interval_2x2` — run the surrogate model with interval estimation and create a 2x2 figure.
- `surrogate_heatmap_r2` - create several heatmaps with values of coefficient of determination.

### Calibration / forecasting
- `calibrate_model_complete_data` — calibrate parameters of a chosen model to a target incidence curve and save plots.
- `calibrate_model_complete_data_3in1` — calibrate parameters of three models (network, hybrid, surrogate) to a target incidence curve and save plots.
- `calibrate_model_forecast` — short-time forecasting; calibrate parameters of a chosen model to an incomplete target incidence curve and save plots.
- `calibrate_model_forecast_3in1` — short-time forecasting; calibrate parameters of three models (network, hybrid, surrogate) to an incomplete target incidence curve and save plots.

### Other plotting tools
- `plot_synth_peaks` — create several heatmaps showing the distribution of peak time and peak incidence for synthetic incidence curves.
- `plot_synth_inc_beta` — create plots showing incidence trajectories and beta (infection transmission rate) trajectories.
- `plot_forecast_peak_errors` - create plots showing peak errors for hybrid and surrogate approaches for short-term forecasting.
- `plot_heatmap_switch` - create several heatmaps showing switching behavior for test samples: difference between epidemic peak time and day of switch; distribution of switch days across all runs.