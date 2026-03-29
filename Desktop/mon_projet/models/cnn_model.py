import tensorflow as tf


class CustomCNN(tf.keras.Model):

    def __init__(self, num_classes=10, **kwargs):
        super(CustomCNN, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
        ])

        # convolution 1
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.bn1   = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))

        # convolution 2
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.bn2   = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))

        # convolution 3
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.bn3   = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))

        # Classifieur final
        self.flatten      = tf.keras.layers.Flatten()
        self.dense1       = tf.keras.layers.Dense(256, activation='relu')
        self.dropout      = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):

        x = self.augmentation(inputs, training=training)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.output_layer(x)

        return x

    def get_config(self):
        config = super(CustomCNN, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        config.pop("trainable", None)
        config.pop("dtype", None)
        return cls(**config)

    def build_graph(self, input_shape=(32, 32, 3)):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))