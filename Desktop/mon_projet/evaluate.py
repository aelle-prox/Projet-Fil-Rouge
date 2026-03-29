import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from models.cnn_model import CustomCNN
from utils.data_loader import load_cifar10, CLASS_NAMES

MODEL_PATH = "saved_model/best_cnn.keras"


def main():

    # ── Etape 1 : Charger les donnees de test uniquement ──
    print("Chargement des donnees...")
    (_, _), (x_test, y_test) = load_cifar10()

    # ── Etape 2 : Charger le modele sauvegarde ──
    print(f"Chargement du modele depuis {MODEL_PATH}...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"CustomCNN": CustomCNN}
    )
    print("Modele charge avec succes !")

    # ── Etape 3 : Generer les predictions ──
    print("Generation des predictions...")
    y_proba = model.predict(x_test, batch_size=64, verbose=1)
    y_pred  = np.argmax(y_proba, axis=1)

    # ── Etape 4 : Score final ──
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")

    # ── Etape 5 : Rapport de classification ──
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # ── Etape 6 : Matrice de confusion ──
    print("Generation de la matrice de confusion...")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matrice de Confusion — CIFAR-10', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe predite')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.show()
    print("Matrice sauvegardee : outputs/confusion_matrix.png")

    # ── Etape 7 : Exemples de predictions ──
    print("Generation des exemples...")
    rng     = np.random.default_rng(42)
    indices = rng.choice(len(x_test), size=16, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        idx    = indices[i]
        ax.imshow(x_test[idx])
        vrai   = CLASS_NAMES[y_test[idx]]
        predit = CLASS_NAMES[y_pred[idx]]
        color  = 'green' if y_test[idx] == y_pred[idx] else 'red'
        ax.set_title(f"Vrai: {vrai}\nPredit: {predit}",
                     fontsize=8, color=color, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Exemples — vert=correct, rouge=erreur', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/sample_predictions.png', dpi=150)
    plt.show()
    print("Exemples sauvegardes : outputs/sample_predictions.png")


if __name__ == "__main__":
    main()