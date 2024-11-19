#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Imports
import lib.hardware as hw
import keras
from keras import layers

strategy = hw.detect_hardware()
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

EPOCHS = 5
BATCH_SIZE = 128 * strategy.num_replicas_in_sync # Gobal batch size.
# The global batch size will be automatically sharded across all
# replicas by the tf.data.Dataset API. A single TPU has 8 cores.
# The best practice is to scale the batch size by the number of
# replicas (cores). The learning rate should be increased as well.

max_features = 20000  # Only consider the top 20k words
maxlen = 300  # Only consider the first 200 words of each movie review

"""
## Load the IMDB movie review sentiment data
"""
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
# Use pad_sequence to standardize sequence length:
# this will truncate sequences longer than 200 words and zero-pad sequences shorter than 200 words.
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)

"""
## Build the model
"""
with strategy.scope():
  # Input for variable-length sequences of integers
  inputs = keras.Input(shape=(None,), dtype="int32")
  # Embed each integer in a 128-dimensional vector
  x = layers.Embedding(max_features, 128)(inputs)
  # Add 2 bidirectional LSTMs
  x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
  x = layers.Bidirectional(layers.LSTM(64))(x)
  # Add a classifier
  outputs = layers.Dense(1, activation="sigmoid")(x)
  model = keras.Model(inputs, outputs)
model.summary()

"""
## Train and evaluate the model

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/bidirectional-lstm-imdb)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/bidirectional_lstm_imdb).
"""
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val))

