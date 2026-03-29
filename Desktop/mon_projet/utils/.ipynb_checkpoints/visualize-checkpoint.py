import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)


def plot_training_history(history):
    h      = history.history
    epochs = range(1, len(h['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Resultats de l'entrainement", fontsize=14, fontweight='bold')

    ax1.plot(epochs, h['loss'],     'b-o', markersize=4, label='Train Loss')
    ax1.plot(epochs, h['val_loss'], 'r-o', markersize=4, label='Val Loss')
    ax1.set_title('Courbe de perte (Loss)')
    ax1.set_xlabel('Epoque')
    ax1.set_ylabel('Perte')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    best = np.argmin(h['val_loss']) + 1
    ax1.axvline(best, color='gray', linestyle='--', alpha=0.7, label=f'Meilleure epoque : {best}')
    ax1.legend()

    ax2.plot(epochs, h['accuracy'],     'b-o', markersize=4, label='Train Accuracy')
    ax2.plot(epochs, h['val_accuracy'], 'r-o', markersize=4, label='Val Accuracy')
    ax2.set_title('Courbe de precision (Accuracy)')
    ax2.set_xlabel('Epoque')
    ax2.set_ylabel('Precision')
    ax2.axhline(0.70, color='green', linestyle='--', alpha=0.7, label='Objectif 70%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Courbes sauvegardees : outputs/training_history.png")


def plot_confusion_matrix(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion — CIFAR-10', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe predite')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.show()
    print("Matrice sauvegardee : outputs/confusion_matrix.png")


def plot_sample_predictions(images, y_true, y_pred, class_names, n=16):
    cols = 4
    rows = n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle('Exemples — vert=correct, rouge=erreur', fontsize=12, fontweight='bold')

    rng     = np.random.default_rng(42)
    indices = rng.choice(len(images), size=n, replace=False)

    for i, ax in enumerate(axes.flat):
        idx     = indices[i]
        ax.imshow(images[idx])
        vrai    = class_names[y_true[idx]]
        predit  = class_names[y_pred[idx]]
        couleur = 'green' if y_true[idx] == y_pred[idx] else 'red'
        ax.set_title(f"Vrai: {vrai}\nPredit: {predit}",
                     fontsize=8, color=couleur, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('outputs/sample_predictions.png', dpi=150)
    plt.show()
    print("Exemples sauvegardes : outputs/sample_predictions.png")


def visualize_data_samples(x_train, y_train, class_names):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Exemples CIFAR-10 — verification des labels", fontsize=13, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[i])
        ax.set_title(f"Label : {class_names[y_train[i]]}\nIndex : {y_train[i]}",
                     fontsize=9, color='darkblue')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('outputs/data_samples.png', dpi=150)
    plt.show()
    print("Exemples sauvegardes : outputs/data_samples.png")


def visualize_augmented_images(x_train):
    import tensorflow as tf

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    image_originale = x_train[0:1]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Image originale vs images augmentees", fontsize=13, fontweight='bold')

    axes.flat[0].imshow(image_originale[0])
    axes.flat[0].set_title("Originale", fontsize=9, color='darkgreen', fontweight='bold')
    axes.flat[0].axis('off')

    for i in range(1, 10):
        img_aug = augmentation(image_originale, training=True)
        axes.flat[i].imshow(img_aug[0].numpy())
        axes.flat[i].set_title(f"Augmentee {i}", fontsize=9, color='darkorange')
        axes.flat[i].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/augmented_images.png', dpi=150)
    plt.show()
    print("Images augmentees sauvegardees : outputs/augmented_images.png")


def verify_label_coherence(y_train, y_test, class_names):
    print("=== Verification de la coherence des labels ===\n")

    train_counts = np.bincount(y_train)
    test_counts  = np.bincount(y_test)

    print(f"{'Classe':<15} {'Train':>8} {'Test':>8} {'Total':>8}")
    print("-" * 42)
    for i, name in enumerate(class_names):
        print(f"{name:<15} {train_counts[i]:>8} {test_counts[i]:>8} {train_counts[i]+test_counts[i]:>8}")
    print("-" * 42)
    print(f"{'TOTAL':<15} {sum(train_counts):>8} {sum(test_counts):>8} {sum(train_counts)+sum(test_counts):>8}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribution des classes", fontsize=13, fontweight='bold')

    ax1.bar(class_names, train_counts, color='steelblue')
    ax1.set_title("Train (50 000 images)")
    ax1.set_xlabel("Classe")
    ax1.set_ylabel("Nombre d'images")
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(5000, color='red', linestyle='--', label='5000 par classe')
    ax1.legend()

    ax2.bar(class_names, test_counts, color='coral')
    ax2.set_title("Test (10 000 images)")
    ax2.set_xlabel("Classe")
    ax2.set_ylabel("Nombre d'images")
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(1000, color='red', linestyle='--', label='1000 par classe')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('outputs/label_distribution.png', dpi=150)
    plt.show()
    print("\nDistribution sauvegardee : outputs/label_distribution.png")
    print("Conclusion : dataset equilibre — chaque classe a le meme nombre d'images")

def plot_lstm_predictions(y_true, y_pred, n_display=500):
    """
    Graphique Temperature reelle vs predite.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(y_true[:n_display], 'b-', linewidth=1,
             label='Temperature reelle (C)', alpha=0.8)
    plt.plot(y_pred[:n_display], 'r-', linewidth=1,
             label='Temperature predite (C)', alpha=0.8)
    plt.title('LSTM — Temperature reelle vs predite', fontsize=13, fontweight='bold')
    plt.xlabel('Pas de temps')
    plt.ylabel('Temperature (degres Celsius)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/lstm_real_vs_predicted.png', dpi=150)
    plt.show()
    print("Graphique sauvegarde : outputs/lstm_real_vs_predicted.png")


def plot_lstm_loss(history):
    """
    Courbes Train Loss vs Val Loss pour le LSTM.
    """
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history.history['loss'],     'b-o', markersize=4, label='Train Loss (MSE)')
    plt.plot(epochs, history.history['val_loss'], 'r-o', markersize=4, label='Val Loss (MSE)')

    best = np.argmin(history.history['val_loss']) + 1
    plt.axvline(best, color='gray', linestyle='--', alpha=0.7,
                label=f'Meilleure epoque : {best}')

    plt.title('Courbes de perte LSTM', fontsize=13, fontweight='bold')
    plt.xlabel('Epoque')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/lstm_training_curves.png', dpi=150)
    plt.show()
    print("Courbes sauvegardees : outputs/lstm_training_curves.png")
    
def plot_lstm_data_overview(temperatures, n_display=1000):
    """
    Visualise un apercu des donnees de temperature brutes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Apercu des donnees meteorologiques", fontsize=13, fontweight='bold')

    # Courbe des temperatures
    ax1.plot(temperatures[:n_display], 'b-', linewidth=0.8, alpha=0.8)
    ax1.set_title(f"Serie temporelle ({n_display} premieres mesures)")
    ax1.set_xlabel("Pas de temps")
    ax1.set_ylabel("Temperature (degres C)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(temperatures.mean(), color='red', linestyle='--',
                label=f"Moyenne : {temperatures.mean():.1f} C")
    ax1.legend()

    # Distribution des temperatures
    ax2.hist(temperatures, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.set_title("Distribution des temperatures")
    ax2.set_xlabel("Temperature (degres C)")
    ax2.set_ylabel("Frequence")
    ax2.axvline(temperatures.mean(), color='red', linestyle='--',
                label=f"Moyenne : {temperatures.mean():.1f} C")
    ax2.axvline(temperatures.min(), color='orange', linestyle='--',
                label=f"Min : {temperatures.min():.1f} C")
    ax2.axvline(temperatures.max(), color='green', linestyle='--',
                label=f"Max : {temperatures.max():.1f} C")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/lstm_data_overview.png', dpi=150)
    plt.show()
    print("Apercu donnees sauvegarde : outputs/lstm_data_overview.png")


def plot_lstm_sliding_window(temperatures, sequence_length=24):
    """
    Visualise le principe du Sliding Window sur un extrait des donnees.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Principe du Sliding Window (fenetre glissante)", fontsize=13, fontweight='bold')

    extrait = temperatures[:sequence_length + 5]
    ax.plot(range(len(extrait)), extrait, 'b-o', markersize=6, linewidth=1.5, label="Serie temporelle")

    # Fenetre 1
    ax.axvspan(0, sequence_length - 1, alpha=0.15, color='blue', label=f"Fenetre entree (24 points)")

    # Point a predire
    ax.scatter([sequence_length], [extrait[sequence_length]], color='red', s=150, zorder=5,
               label=f"Valeur a predire (T+1)")

    ax.set_xlabel("Pas de temps")
    ax.set_ylabel("Temperature (degres C)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/lstm_sliding_window.png', dpi=150)
    plt.show()
    print("Sliding window sauvegarde : outputs/lstm_sliding_window.png")