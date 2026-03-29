import tensorflow as tf
from utils.visualize import plot_training_history
from models.cnn_model import CustomCNN
from utils.data_loader import load_cifar10, create_datasets

BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 0.001
MODEL_PATH    = "saved_model/best_cnn.keras"


def main():

    # ── Etape 1 : Chargement des donnees ──
    print("Chargement des donnees CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    train_ds, val_ds, test_ds = create_datasets(
        x_train, y_train, x_test, y_test, BATCH_SIZE
    )

    # ── Etape 2 : Creation du modele ──
    print("Creation du modele...")
    model = CustomCNN(num_classes=10)
    model.build_graph().summary()

    # ── Etape 3 : Compilation ──
    print("Compilation du modele...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    print("Modele compile avec succes !")

    # ── Etape 4 : Callbacks ──
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # ── Etape 5 : Entrainement ──
    print("Lancement de l'entrainement...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    print("Entrainement termine !")
    plot_training_history(history)

    # ── Etape 6 : Evaluation finale ──
    print("Evaluation sur le jeu de test...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")

    if test_acc >= 0.70:
        print("OBJECTIF ATTEINT — accuracy >= 70% !")
    else:
        print("Objectif non atteint — continuer a optimiser.")


if __name__ == "__main__":
    main()