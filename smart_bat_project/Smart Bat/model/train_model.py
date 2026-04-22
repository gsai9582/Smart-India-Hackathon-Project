import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

DATA_FILE = "../data_collection/training_data.csv"
MODEL_FILE = "bat_model.pkl"

def load_and_prepare():
    df = pd.read_csv(DATA_FILE)

    # Drop unlabeled rows
    df = df[df["label"] != "none"]

    features = ["ax", "ay", "az", "gx", "gy", "gz", "piezo"]
    X = df[features].astype(float)
    y = df["label"]

    return X, y

def train():
    X, y = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    Path(MODEL_FILE).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print("Saved model to", MODEL_FILE)

if __name__ == "__main__":
    train()
