
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph"]
TARGET_COLUMN = "label"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_data(csv_path: str) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required column(s) in dataset: {missing_cols}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )
    print(f"[OK] All required columns present: {REQUIRED_COLUMNS}")


def validate_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    invalid_report = {}

    for col in FEATURE_COLUMNS:
        
        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")

        newly_invalid_mask = converted.isna() & original.notna()
        n_invalid = int(newly_invalid_mask.sum())

        if n_invalid > 0:
            invalid_report[col] = {
                "count": n_invalid,
                "example_values": original[newly_invalid_mask].unique()[:5].tolist(),
            }

        df[col] = converted

    if invalid_report:
        print("[WARNING] Invalid (non-numeric) sensor values detected and converted to NaN:")
        for col, info in invalid_report.items():
         print(f"  - {col}: {info['count']} invalid value(s), examples: {info['example_values']}")
        print("[OK] All sensor feature columns are numeric.")

    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    
    n_before = len(df)
    n_duplicates = df.duplicated().sum()

    if n_duplicates > 0:
        print(f"[WARNING] Found {n_duplicates} duplicate row(s). Removing them.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[OK] No duplicate rows found.")

    n_after = len(df)
    print(f"    Rows before: {n_before}, Rows after: {n_after}")
    return df


def handle_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    
    n_missing = df[TARGET_COLUMN].isna().sum()

    if n_missing > 0:
        print(f"[WARNING] Found {n_missing} row(s) with missing '{TARGET_COLUMN}'. Dropping them.")
        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    else:
        print(f"[OK] No missing values in target column '{TARGET_COLUMN}'.")

    return df


def encode_target(y: pd.Series) -> tuple:
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("[OK] Label encoding mapping:")
    for class_name, class_id in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"  {class_name} -> {class_id}")

    return y_encoded, label_encoder


def preprocess_pipeline(csv_path: str):
    
    df = load_data(csv_path)

    validate_columns(df)

    df = validate_numeric_features(df)

    df = handle_duplicates(df)

    df = handle_missing_target(df)

    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    y_encoded, label_encoder = encode_target(y)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )
    print(f"[OK] Split data: {len(X_train)} train rows, {len(X_test)} test rows.")

    
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=FEATURE_COLUMNS,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),  
        columns=FEATURE_COLUMNS,
        index=X_test.index
    )
    print("[OK] Missing feature values imputed using median strategy (fit on training data only).")

    
    return X_train_imputed, X_test_imputed, y_train, y_test, imputer, label_encoder


def preprocess_live_reading(sensor_values: dict, imputer: SimpleImputer) -> pd.DataFrame:
    
    missing_keys = [col for col in FEATURE_COLUMNS if col not in sensor_values]
    if missing_keys:
        raise ValueError(f"Live sensor reading is missing required field(s): {missing_keys}")

    row = {}
    for col in FEATURE_COLUMNS:
        value = pd.to_numeric(sensor_values[col], errors="coerce")
        row[col] = value

    df_row = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    if df_row.isna().any().any():
        print("[WARNING] Live reading contained invalid/missing value(s); imputing with training median.")

    df_row_imputed = pd.DataFrame(
        imputer.transform(df_row),
        columns=FEATURE_COLUMNS
    )
    return df_row_imputed


if __name__ == "__main__":
    
    CSV_PATH = "data/soil_data.csv"

    X_train, X_test, y_train, y_test, imputer, label_encoder = preprocess_pipeline(CSV_PATH)

    print("\n--- Preprocessing Summary ---")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Feature order:", FEATURE_COLUMNS)
    print("Classes:", list(label_encoder.classes_))
