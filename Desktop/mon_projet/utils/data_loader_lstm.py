import numpy as np
import tensorflow as tf
import pandas as pd
import os
from zipfile import ZipFile


def load_jena_climate():
    """
    Charge le dataset meteo de Jena depuis Keras (URL .zip correcte).
    """
    uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

    try:
        print("Telechargement du dataset Jena Climate...")
        zip_path = tf.keras.utils.get_file(
            origin=uri,
            fname="jena_climate_2009_2016.csv.zip",
            extract=True,
        )
        csv_path = zip_path.replace(".zip", "")

        if not os.path.exists(csv_path):
            zip_file = ZipFile(zip_path)
            zip_file.extractall(os.path.dirname(zip_path))

        df = pd.read_csv(csv_path)
        temperatures = df["T (degC)"].values.astype(np.float32)

        # Sauvegarder dans data/
        os.makedirs("data", exist_ok=True)
        np.save("data/temperatures_jena.npy", temperatures)
        print(f"Donnees sauvegardees dans data/temperatures_jena.npy")
        print(f"Nombre de mesures : {len(temperatures)}")
        print(f"Temperature min: {temperatures.min():.1f}C | max: {temperatures.max():.1f}C")
        return temperatures

    except Exception as e:
        print(f"Telechargement echoue : {e}")
        print("Utilisation des donnees simulees...")
        return generate_simulated_weather()


def generate_simulated_weather():
    """
    Genere des donnees de temperature simulees realistes.
    """
    np.random.seed(42)
    n_points = 100_000
    t = np.linspace(0, 4 * np.pi, n_points)
    annual = 15 * np.sin(t) + 10
    daily  = 3  * np.sin(t * 365 / 2)
    noise  = np.random.normal(0, 0.5, n_points)
    temperatures = (annual + daily + noise).astype(np.float32)

    os.makedirs("data", exist_ok=True)
    np.save("data/temperatures_simulees.npy", temperatures)
    print(f"Donnees simulees sauvegardees dans data/temperatures_simulees.npy")
    print(f"Nombre de mesures : {len(temperatures)}")
    print(f"Temperature min: {temperatures.min():.1f}C | max: {temperatures.max():.1f}C")
    return temperatures


def normalize_data(data):
    data_min = data.min()
    data_max = data.max()
    data_norm = (data - data_min) / (data_max - data_min)
    print(f"Normalisation : min={data_min:.2f} | max={data_max:.2f}")
    return data_norm, data_min, data_max


def inverse_normalize(data_norm, data_min, data_max):
    return data_norm * (data_max - data_min) + data_min


def create_sequences(data, sequence_length=24, train_ratio=0.8):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split   = int(len(X) * train_ratio)
    x_train = X[:split];  x_test  = X[split:]
    y_train = y[:split];  y_test  = y[split:]

    print(f"\nSequences creees :")
    print(f"  Train : {x_train.shape} | Labels : {y_train.shape}")
    print(f"  Test  : {x_test.shape}  | Labels : {y_test.shape}")
    return x_train, y_train, x_test, y_test


def create_lstm_datasets(x_train, y_train, x_test, y_test, batch_size=32):
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .cache().shuffle(len(x_train)).batch(batch_size).prefetch(AUTOTUNE))
    test_ds  = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                .cache().batch(batch_size).prefetch(AUTOTUNE))
    print(f"Datasets LSTM crees (batch_size={batch_size})")
    return train_ds, test_ds
