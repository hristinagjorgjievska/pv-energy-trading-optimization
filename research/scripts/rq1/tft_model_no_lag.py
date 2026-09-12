import pandas as pd
import numpy as np
import lightning.pytorch as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
all_results = {}

if __name__ == "__main__":
    for country in COUNTRIES:
        df = pd.read_csv(f"{country}_features.csv")
        df["time_idx"] = np.arange(len(df))
        df["group"] = country

        target = "Price (EUR/MWhe)"
        features = [
            'temperature_2m', 'wind_speed_10m', 'cloud_cover', 'shortwave_radiation',
        ]

        max_encoder_length = 168
        max_prediction_length = 24

        train_end = int(len(df) * 0.7)
        training_cutoff = df["time_idx"].iloc[train_end]

        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=target,
            group_ids=["group"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=["hour", "day_of_week", "month", "is_weekend"],
            time_varying_unknown_reals=[target] + features,
        )

        test_data = df[lambda x: x.time_idx > training_cutoff - max_encoder_length]

        validation = TimeSeriesDataSet.from_dataset(
            training, test_data, predict=False, stop_randomization=True
        )

        train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=2)
        val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=2)

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.001,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            loss=QuantileLoss(),
        )

        trainer = pl.Trainer(max_epochs=20, accelerator="gpu", gradient_clip_val=0.1)
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        predictions = tft.predict(val_dataloader, mode="prediction")
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])

        y_pred = predictions.numpy().flatten()
        y_true = actuals.numpy().flatten()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"{country}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2*100:.1f}%")
        all_results[country] = {"MAE": mae, "RMSE": rmse, "R2": r2 * 100}

        pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2 * 100}]).to_csv(f"tft_no_lag_{country}.csv", index=False)

        results_df = pd.DataFrame(all_results).T
        results_df.to_csv("tft_no_lag_model.csv")
        print(results_df)
