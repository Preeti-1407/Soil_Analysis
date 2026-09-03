
import os
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_preprocessing import preprocess_pipeline


DATA_PATH = "data/soil_data.csv"
MODELS_DIR = "models"

MODEL_PATH = os.path.join(MODELS_DIR, "soil_quality_model.pkl")
IMPUTER_PATH = os.path.join(MODELS_DIR, "imputer.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


def main():

    # Check dataset
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at '{DATA_PATH}'.")
        sys.exit(1)

    # Preprocess dataset
    try:
        X_train, X_test, y_train, y_test, imputer, label_encoder = (
            preprocess_pipeline(DATA_PATH)
        )
    except Exception as e:
        print(f"ERROR: Preprocessing failed: {e}")
        sys.exit(1)

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("\n===== Evaluation Results =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Create models folder
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model and preprocessing objects
    joblib.dump(model, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print("\nModel files saved successfully.")
    print(f"Model:   {MODEL_PATH}")
    print(f"Imputer: {IMPUTER_PATH}")
    print(f"Encoder: {ENCODER_PATH}")


if __name__ == "__main__":
    main()
