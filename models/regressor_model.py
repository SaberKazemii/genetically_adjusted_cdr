#!/usr/bin/env python
# coding: utf-8

# In[1]:


from test_model import ClassifierModel
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

# Assuming R2Score is either defined elsewhere or imported
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.sst = self.add_weight(name="sst", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = y_true - y_pred
        self.sse.assign_add(tf.reduce_sum(tf.square(error)))
        mean_y = tf.reduce_mean(y_true)
        total_error = y_true - mean_y
        self.sst.assign_add(tf.reduce_sum(tf.square(total_error)))

    def result(self):
        return 1 - (self.sse / self.sst)

    def reset_states(self):
        self.sse.assign(0)
        self.sst.assign(0)

class ModifiedClassifierModel(ClassifierModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base ClassifierModel
        self.new_model = self._modify_model()  # Modify the model as required

    def _modify_model(self):
        # Freeze the layers except the last one
        for layer in self.model.layers[:-1]:
            layer.trainable = False

        # Remove the last layer and keep all layers up to the second-to-last
        model_without_last_layer = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

        # Add a new Dense layer with linear activation as the only trainable layer
        new_output = Dense(1, activation='linear', name='output')(model_without_last_layer.output)

        # Create a new model with the modified output layer
        new_model = Model(inputs=model_without_last_layer.input, outputs=new_output)

        # Compile the new model with a different optimizer, loss function, and metrics
        new_model.compile(optimizer='adam', loss='mean_squared_error', metrics=[R2Score()])

        return new_model

