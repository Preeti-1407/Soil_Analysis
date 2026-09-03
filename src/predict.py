
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np


FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph"]

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "soil_quality_model.pkl"
IMPUTER_PATH = MODEL_DIR / "imputer.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"


class ModelLoadError(Exception):
    pass


class InvalidInputError(Exception):
    pass


class PredictionError(Exception):
    pass


def load_artifacts():
    paths = {
        "model": MODEL_PATH,
        "imputer": IMPUTER_PATH,
        "label_encoder": LABEL_ENCODER_PATH
    }

    artifacts = {}

    for name, path in paths.items():
        if not path.exists():
            raise ModelLoadError(f"{name} file not found: {path}")

        try:
            artifacts[name] = joblib.load(path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load {name}: {exc}"
            ) from exc

    return (
        artifacts["model"],
        artifacts["imputer"],
        artifacts["label_encoder"]
    )


def validate_inputs(N, P, K, temperature, humidity, ph):

    values = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph
    }

    cleaned = []

    for name, value in values.items():

        if value is None:
            raise InvalidInputError(f"Missing value for '{name}'.")

        if isinstance(value, bool):
            raise InvalidInputError(f"'{name}' must be numeric.")

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise InvalidInputError(
                f"'{name}' must be numeric."
            )

        if np.isnan(value):
            raise InvalidInputError(f"'{name}' cannot be NaN.")

        cleaned.append(value)

    return cleaned


def predict_soil_quality(N, P, K, temperature, humidity, ph):

    model, imputer, label_encoder = load_artifacts()

    values = validate_inputs(
        N, P, K, temperature, humidity, ph
    )

    input_df = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )

    try:
        input_df = pd.DataFrame(
            imputer.transform(input_df),
            columns=FEATURE_COLUMNS
        )

        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        predicted_label = label_encoder.inverse_transform(
            prediction
        )[0]

    except Exception as exc:
        raise PredictionError(
            f"Prediction failed: {exc}"
        ) from exc

    class_labels = label_encoder.classes_

    probability_dict = {
        label: float(prob)
        for label, prob in zip(
            class_labels,
            probabilities[0]
        )
    }

    return {
        "prediction": predicted_label,
        "probabilities": probability_dict
    }


def print_prediction_report(reading, result):

    print("\nNew Soil Reading:")
    
    for name, value in reading.items():
        print(f"{name}: {value}")

    print(f"\nPredicted Soil Quality: {result['prediction']}\n")

    for label, probability in result["probabilities"].items():
        print(f"{label}: {probability * 100:.2f}%")


if __name__ == "__main__":

    manual_reading = {
        "N": 30,
        "P": 25,
        "K": 50,
        "temperature": 80,
        "humidity": 32,
        "ph": 5.5
    }

    try:
        result = predict_soil_quality(**manual_reading)
        print_prediction_report(manual_reading, result)

    except ModelLoadError as e:
        print(f"[Model Load Error] {e}", file=sys.stderr)
        sys.exit(1)

    except InvalidInputError as e:
        print(f"[Invalid Input] {e}", file=sys.stderr)
        sys.exit(1)

    except PredictionError as e:
        print(f"[Prediction Error] {e}", file=sys.stderr)
        sys.exit(1)
