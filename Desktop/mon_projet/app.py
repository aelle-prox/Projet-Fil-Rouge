import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io
import os

from models.cnn_model import CustomCNN
from models.rnn_model import CustomLSTM
from utils.data_loader_lstm import normalize_data, generate_simulated_weather

app = Flask(__name__)

# ── Noms des classes CIFAR-10 ──────────────────────────────────────────────
CLASS_NAMES = [
    'avion', 'automobile', 'oiseau', 'chat', 'cerf',
    'chien', 'grenouille', 'cheval', 'bateau', 'camion'
]

# ── Chargement des modeles au demarrage ────────────────────────────────────
print("Chargement des modeles...")

CNN_MODEL = tf.keras.models.load_model(
    "saved_model/best_cnn.keras",
    custom_objects={"CustomCNN": CustomCNN}
)
print("CNN charge !")

LSTM_MODEL = tf.keras.models.load_model(
    "saved_model/best_lstm.keras",
    custom_objects={"CustomLSTM": CustomLSTM}
)
print("LSTM charge !")

# Min/Max pour la normalisation LSTM
data_complete      = generate_simulated_weather()
_, DATA_MIN, DATA_MAX = normalize_data(data_complete)
print("API prete !")


# ── Route principale ───────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API Deep Learning — CNN + LSTM",
        "routes": {
            "/predict/image": "POST — Classification d'image (CNN)",
            "/predict/temperature": "POST — Prediction temperature (LSTM)",
            "/health": "GET — Statut de l'API"
        }
    })


# ── Route de sante ─────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "cnn_model": "charge",
        "lstm_model": "charge",
        "classes": CLASS_NAMES
    })


# ── Route CNN — Classification d'image ────────────────────────────────────
@app.route("/predict/image", methods=["POST"])
def predict_image():
    """
    Recoit une image et retourne la classe predite.
    Envoyer avec : curl -X POST -F "image=@mon_chat.jpg" http://localhost:5000/predict/image
    """
    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image envoyee — utilisez le champ 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"erreur": "Fichier vide"}), 400

    try:
        # ── Pretraitement obligatoire ──────────────────────────────────────
        # 1. Lire l'image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # 2. Convertir en RGB (elimine le canal alpha si present)
        img = img.convert("RGB")

        # 3. Redimensionner en 32x32 (taille CIFAR-10)
        img = img.resize((32, 32))

        # 4. Convertir en tableau NumPy
        img_array = np.array(img, dtype=np.float32)

        # 5. Normaliser entre 0 et 1
        img_array = img_array / 255.0

        # 6. Ajouter la dimension batch : (32,32,3) → (1,32,32,3)
        img_array = np.expand_dims(img_array, axis=0)

        # ── Prediction ────────────────────────────────────────────────────
        predictions    = CNN_MODEL.predict(img_array, verbose=0)
        predicted_idx  = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence     = float(predictions[0][predicted_idx]) * 100

        # Toutes les probabilites
        all_probs = {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "classe_predite": predicted_class,
            "confiance": f"{confidence:.2f}%",
            "toutes_les_probabilites": all_probs
        })

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


# ── Route LSTM — Prediction de temperature ─────────────────────────────────
@app.route("/predict/temperature", methods=["POST"])
def predict_temperature():
    """
    Recoit 24 temperatures et predit la suivante.
    Envoyer avec JSON :
    {
        "temperatures": [10.2, 10.5, 11.0, ..., 16.8]  <- 24 valeurs
    }
    """
    data = request.get_json()

    if not data or "temperatures" not in data:
        return jsonify({
            "erreur": "Envoyer un JSON avec le champ 'temperatures' contenant 24 valeurs"
        }), 400

    temperatures = data["temperatures"]

    if len(temperatures) != 24:
        return jsonify({
            "erreur": f"Il faut exactement 24 temperatures — vous en avez envoye {len(temperatures)}"
        }), 400

    try:
        # ── Pretraitement ──────────────────────────────────────────────────
        temps_array = np.array(temperatures, dtype=np.float32)

        # Normalisation MinMax
        temps_norm = (temps_array - DATA_MIN) / (DATA_MAX - DATA_MIN)

        # Reshape pour LSTM : (1, 24, 1)
        x = temps_norm.reshape(1, 24, 1)

        # ── Prediction ────────────────────────────────────────────────────
        pred_norm = float(LSTM_MODEL.predict(x, verbose=0)[0][0])

        # Inverser la normalisation
        pred_real = pred_norm * (DATA_MAX - DATA_MIN) + DATA_MIN

        return jsonify({
            "temperatures_entrees": temperatures,
            "temperature_predite": round(float(pred_real), 2),
            "unite": "degres Celsius",
            "horizon": "T+1 (prochaine mesure)"
        })

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


# ── Lancement ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("API Deep Learning demarre sur http://localhost:5000")
    print("="*50)
    app.run(debug=True, host="0.0.0.0", port=5000)