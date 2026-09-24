import argparse
import json
import os

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Bio-Rad x/y gates from a feature CSV."
    )
    parser.add_argument("data_path")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    with open(args.metrics_path, "r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    feature_columns = metrics["feature_columns"]
    missing_columns = [column for column in feature_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Required feature columns are missing: {', '.join(missing_columns)}")

    model = joblib.load(args.model_path)
    predictions = model.predict(data[feature_columns])
    result = data.copy()
    result["predicted_x_gate"] = predictions[:, 0]
    result["predicted_y_gate"] = predictions[:, 1]

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")
    print(f"Predicted wells: {len(result)}")


if __name__ == "__main__":
    main()