import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_ROWS = 5000
NUMERIC_COLUMNS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
INTEGER_COLUMNS = ["Age", "RestingBP", "Cholesterol", "MaxHR"]
BOUNDS = {
    "Age": (18, 100),
    "RestingBP": (1, 250),
    "Cholesterol": (1, 700),
    "MaxHR": (1, 250),
    "Oldpeak": (-5, 10),
}


def encode(data):
    encoded = pd.get_dummies(data, drop_first=True).astype(int)
    return encoded.drop(columns=["HeartDisease"], errors="ignore")


def make_synthetic_rows(training_data, count, rng):
    source = training_data.sample(n=count, replace=True, random_state=RANDOM_STATE).reset_index(drop=True)
    synthetic = source.copy()
    noise = {
        "Age": 2.0,
        "RestingBP": 8.0,
        "Cholesterol": 20.0,
        "MaxHR": 8.0,
        "Oldpeak": 0.35,
    }
    for column in NUMERIC_COLUMNS:
        values = synthetic[column].astype(float).to_numpy()
        values += rng.normal(0, noise[column], size=count)
        minimum, maximum = BOUNDS[column]
        values = np.clip(values, minimum, maximum)
        synthetic[column] = np.rint(values).astype(int) if column in INTEGER_COLUMNS else np.round(values, 1)
    return synthetic


def main():
    original = pd.read_csv("heart.csv")
    if len(original) >= TARGET_ROWS:
        raise ValueError(f"heart.csv already contains {len(original)} rows.")

    train_data, test_data = train_test_split(
        original,
        test_size=0.2,
        stratify=original["HeartDisease"],
        random_state=RANDOM_STATE,
    )
    rng = np.random.default_rng(RANDOM_STATE)
    synthetic_count = TARGET_ROWS - len(original)
    synthetic = make_synthetic_rows(train_data, synthetic_count, rng)
    augmented = pd.concat([original, synthetic], ignore_index=True)
    augmented.to_csv("heart.csv", index=False)
    augmented.to_csv("heart_augmented.csv", index=False)

    original_encoded = pd.get_dummies(original, drop_first=True).astype(int)
    x_original = original_encoded.drop(columns=["HeartDisease"])
    y_original = original_encoded["HeartDisease"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_original,
        y_original,
        test_size=0.2,
        stratify=y_original,
        random_state=RANDOM_STATE,
    )
    baseline_scaler = StandardScaler()
    baseline_model = KNeighborsClassifier()
    baseline_model.fit(baseline_scaler.fit_transform(x_train), y_train)
    baseline_prediction = baseline_model.predict(baseline_scaler.transform(x_test))

    encoded = pd.get_dummies(augmented, drop_first=True).astype(int)
    x_augmented = encoded.drop(columns=["HeartDisease"])
    y_augmented = encoded["HeartDisease"]
    scaler = StandardScaler()
    model = KNeighborsClassifier()
    model.fit(scaler.fit_transform(x_augmented), y_augmented)
    augmented_prediction = model.predict(scaler.transform(x_test))

    joblib.dump(model, "knn_heart_model.pkl")
    joblib.dump(scaler, "heart_scaler.pkl")
    joblib.dump(x_augmented.columns.tolist(), "heart_columns.pkl")

    print(f"Original rows: {len(original)}")
    print(f"Synthetic rows added: {synthetic_count}")
    print(f"Augmented rows: {len(augmented)}")
    print(f"Baseline accuracy: {accuracy_score(y_test, baseline_prediction):.4f}")
    print(f"Augmented accuracy: {accuracy_score(y_test, augmented_prediction):.4f}")
    print(f"Baseline F1: {f1_score(y_test, baseline_prediction):.4f}")
    print(f"Augmented F1: {f1_score(y_test, augmented_prediction):.4f}")


if __name__ == "__main__":
    main()