
```markdown name=docs/api_reference.md
# API Reference

## src/data_preprocessing.py

- `load_data(filepath)`: Load inventory data from file.
- `clean_data(df)`: Clean and preprocess data.
- `engineer_features(df)`: Create features.

## src/forecasting_engine.py

- `train_models(X_train, y_train)`: Train forecasting models.
- `predict(models, X_test)`: Predict demand.

## src/inventory_optimizer.py

- `calculate_reorder_point(demand, lead_time, service_level, std_dev)`: Compute reorder point.
- `abc_analysis(df)`: Perform ABC inventory analysis.

## src/visualization.py

- `plot_forecast(prediction_result)`: Plot forecast with intervals.
- `plot_inventory_levels(df)`: Plot inventory levels.

## src/monitoring.py

- `calculate_metrics(y_true, y_pred)`: Forecast accuracy metrics.
- `generate_alerts(inventory_df)`: Generate stock alerts.