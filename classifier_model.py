#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform


class ClassifierModel:
    def __init__(self, learning_rate=0.0001, dropout_rate=0.3, seed_value=42):
        tf.random.set_seed(seed_value)
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.seed_value = seed_value
        self.model = self._build_model()

    def _build_model(self):
        cdr_input = Input(shape=(1,), name='cdr_input')
        prs_input = Input(shape=(1,), name='prs_input')

        combined = tf.keras.layers.concatenate([cdr_input, prs_input])

        x = Dense(64, kernel_initializer=GlorotUniform(seed=self.seed_value))(combined)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(64, kernel_initializer=GlorotUniform(seed=self.seed_value))(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(32, kernel_initializer=GlorotUniform(seed=self.seed_value))(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        output = Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform(seed=self.seed_value))(x)

        model = Model(inputs=[cdr_input, prs_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

        return model

    def get_model(self):
        return self.model






