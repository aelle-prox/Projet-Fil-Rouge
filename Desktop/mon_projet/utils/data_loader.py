import tensorflow as tf
import numpy as np

CLASS_NAMES = [
    'avion', 'automobile', 'oiseau', 'chat', 'cerf',
    'chien', 'grenouille', 'cheval', 'bateau', 'camion'
]


def load_cifar10():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation : pixels de [0, 255] vers [0.0, 1.0]
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32')  / 255.0

    # Aplatir les labels : (50000, 1) vers (50000,)
    y_train = y_train.flatten()
    y_test  = y_test.flatten()

    print(f"Train : {x_train.shape} | Test : {x_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def create_datasets(x_train, y_train, x_test, y_test, batch_size=64):

    AUTOTUNE = tf.data.AUTOTUNE

    # Separation train / validation (90% / 10%)
    n_val   = int(len(x_train) * 0.1)
    x_val   = x_train[:n_val]
    y_val   = y_train[:n_val]
    x_train = x_train[n_val:]
    y_train = y_train[n_val:]

    # Pipeline train
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.cache().shuffle(len(x_train)).batch(batch_size).prefetch(AUTOTUNE)

    # Pipeline validation
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.cache().batch(batch_size).prefetch(AUTOTUNE)

    # Pipeline test
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.cache().batch(batch_size).prefetch(AUTOTUNE)

    print(f"Train : {len(x_train)} | Val : {n_val} | Test : {len(x_test)}")

    return train_ds, val_ds, test_ds