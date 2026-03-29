import numpy as np
import tensorflow as tf
import os

from models.rnn_model import CustomLSTM
from utils.data_loader_lstm import (
    load_jena_climate, normalize_data,
    inverse_normalize, create_sequences,
    create_lstm_datasets
)

from utils.visualize import plot_lstm_predictions, plot_lstm_loss, plot_lstm_data_overview, plot_lstm_sliding_window

os.makedirs("saved_model", exist_ok=True)
os.makedirs("outputs",     exist_ok=True)

SEQUENCE_LENGTH = 24
BATCH_SIZE      = 32
EPOCHS          = 30
LSTM_UNITS      = 64
MODEL_PATH      = "saved_model/best_lstm.keras"


def main():

    # ── Etape 1 : Charger les donnees ──
    print("=" * 50)
    print("ETAPE 1 : Chargement des donnees Jena")
    print("=" * 50)
    temperatures = load_jena_climate()
    # Visualisation des donnees
    plot_lstm_data_overview(temperatures)
    plot_lstm_sliding_window(temperatures)
    
    # ── Etape 2 : Normaliser ──
    print("\n" + "=" * 50)
    print("ETAPE 2 : Normalisation")
    print("=" * 50)
    data_norm, data_min, data_max = normalize_data(temperatures)

    # ── Etape 3 : Creer les sequences ──
    print("\n" + "=" * 50)
    print("ETAPE 3 : Creation des sequences (Sliding Window)")
    print("=" * 50)
    x_train, y_train, x_test, y_test = create_sequences(
        data_norm, sequence_length=SEQUENCE_LENGTH
    )

    # ── Etape 4 : Pipelines tf.data ──
    train_ds, test_ds = create_lstm_datasets(
        x_train, y_train, x_test, y_test, BATCH_SIZE
    )

    # ── Etape 5 : Creer le modele ──
    print("\n" + "=" * 50)
    print("ETAPE 4 : Creation du modele LSTM")
    print("=" * 50)
    model = CustomLSTM(units=LSTM_UNITS, num_features=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    print("Modele compile avec succes !")

    # ── Etape 6 : Callbacks ──
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor='val_loss',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    # ── Etape 7 : Entrainement ──
    print("\n" + "=" * 50)
    print("ETAPE 5 : Entrainement")
    print("=" * 50)
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=callbacks,
        verbose=1
    )
    print("Entrainement termine !")

    # ── Etape 8 : Courbes de loss ──
    plot_lstm_loss(history)

    # ── Etape 9 : Predictions et graphique reel vs predit ──
    print("\n" + "=" * 50)
    print("ETAPE 6 : Predictions et visualisation")
    print("=" * 50)

    y_pred_norm = model.predict(x_test, verbose=0).flatten()

    # Inverser la normalisation
    y_pred_real = inverse_normalize(y_pred_norm, data_min, data_max)
    y_test_real = inverse_normalize(y_test,      data_min, data_max)

    # Graphique reel vs predit
    plot_lstm_predictions(y_test_real, y_pred_real)

    # ── Etape 10 : Metriques finales ──
    mse  = np.mean((y_pred_real - y_test_real) ** 2)
    mae  = np.mean(np.abs(y_pred_real - y_test_real))
    rmse = np.sqrt(mse)

    print(f"\nResultats finaux :")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f} degres Celsius")
    print(f"\nModele sauvegarde : {MODEL_PATH}")
    print("Lancez maintenant : python evaluate_lstm.py")


if __name__ == "__main__":
    main()
    