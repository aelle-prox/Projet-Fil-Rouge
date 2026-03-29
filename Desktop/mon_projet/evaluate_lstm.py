import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.rnn_model import CustomLSTM
from utils.data_loader_lstm import (
    load_jena_climate, normalize_data,
    inverse_normalize, create_sequences
)

MODEL_PATH = "saved_model/best_lstm.keras"


def main():

    # ── Charger les donnees ──
    print("Chargement des donnees...")
    temperatures = load_jena_climate()
    data_norm, data_min, data_max = normalize_data(temperatures)
    _, _, x_test, y_test = create_sequences(data_norm, sequence_length=24)

    # ── Charger le modele ──
    print(f"Chargement du modele depuis {MODEL_PATH}...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"CustomLSTM": CustomLSTM}
    )
    print("Modele charge avec succes !")

    # ── Predictions ──
    print("Generation des predictions...")
    y_pred_norm = model.predict(x_test, verbose=1).flatten()

    # Inverser la normalisation
    y_pred_real = inverse_normalize(y_pred_norm, data_min, data_max)
    y_test_real = inverse_normalize(y_test,      data_min, data_max)

    # ── Metriques ──
    mse  = np.mean((y_pred_real - y_test_real) ** 2)
    mae  = np.mean(np.abs(y_pred_real - y_test_real))
    rmse = np.sqrt(mse)

    print(f"\nResultats evaluation :")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f} degres Celsius")

    # ── Graphique reel vs predit ──
    n_display = 500
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_real[:n_display], 'b-', linewidth=1, label='Reelle', alpha=0.8)
    plt.plot(y_pred_real[:n_display], 'r-', linewidth=1, label='Predite', alpha=0.8)
    plt.title('LSTM — Temperature reelle vs predite', fontsize=13, fontweight='bold')
    plt.xlabel('Pas de temps')
    plt.ylabel('Temperature (degres Celsius)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/lstm_evaluation.png', dpi=150)
    plt.show()
    print("Graphique sauvegarde : outputs/lstm_evaluation.png")


if __name__ == "__main__":
    main()