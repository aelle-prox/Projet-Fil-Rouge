import tensorflow as tf


class CustomLSTM(tf.keras.Model):

    def __init__(self, units=64, num_features=1, **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)

        self.units       = units
        self.num_features = num_features

        # Couche LSTM principale
        self.lstm1 = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True,   # retourne toutes les sorties
            name="lstm1"
        )

        # Deuxieme couche LSTM
        self.lstm2 = tf.keras.layers.LSTM(
            units=units // 2,
            return_sequences=False,  # retourne seulement le dernier etat
            name="lstm2"
        )

        self.dropout = tf.keras.layers.Dropout(0.2, name="dropout")

        # Couche de sortie — predit T+1
        self.output_layer = tf.keras.layers.Dense(1, name="output")

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout(x, training=training)
        x = self.lstm2(x)
        x = self.output_layer(x)
        return x

    def get_config(self):
        config = super(CustomLSTM, self).get_config()
        config.update({
            "units": self.units,
            "num_features": self.num_features
        })
        return config

    @classmethod
    def from_config(cls, config):
        config.pop("trainable", None)
        config.pop("dtype", None)
        return cls(**config)