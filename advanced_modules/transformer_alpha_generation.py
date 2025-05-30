import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoConfig

class TimeSeriesTransformer:
    def __init__(self, input_size=10, output_size=1, num_layers=6, d_model=64):
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.model = self._build_model()
    
    def _build_model(self):
        inputs = tf.keras.Input(shape=(None, self.input_size))
        
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        for _ in range(self.num_layers):
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=self.d_model//8
            )(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attention)
            ffn = tf.keras.layers.Dense(self.d_model*2, activation='relu')(x)
            ffn = tf.keras.layers.Dense(self.d_model)(ffn)
            x = tf.keras.layers.LayerNormalization()(x + ffn)
        
        outputs = tf.keras.layers.Dense(self.output_size)(x)
        outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def train(self, X_train, y_train, epochs=50):
        self.model.compile(optimizer='adam', loss='mse')
        return self.model.fit(X_train, y_train, epochs=epochs, verbose=0)
