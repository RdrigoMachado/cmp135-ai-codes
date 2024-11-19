#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Imports
import datetime
import logging
import logging.handlers
import numpy as np
import pandas as pd
import lib.hardware as hw
import lib.utils as util

from tensorflow.keras.layers import Dense, Input, Embedding, LSTM
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from typing import Tuple

# Hardware detection and setup
strategy = hw.detect_hardware()
REPLICAS = strategy.num_replicas_in_sync
print(f'Using {REPLICAS} replicas')

BATCH_SIZE = 128 * REPLICAS  # Global batch size
SEED = 7
np.random.seed(SEED)

# Configuration constants
URL_MODEL_DATA_PATH = 'https://www.inf.ufrgs.br/~cakunas/data/imdb.csv'
MAX_FEATURES = 20000
MAXLEN = 300
TEST_DIM = 0.20
EMBED_DIM = 128
EPOCHS = 1
LOG_FILENAME = 'sample.log'
MODEL_FILE_PATH = f'tf-model-keras-trained-{datetime.datetime.now():%Y-%m-%d}.keras'

# Initialize logger
def setup_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=256 * 1024 * 1024, backupCount=50, encoding='utf8'
    )
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

logger = setup_logger('SentimentAnalysisLSTM', LOG_FILENAME)
logger.info('Starting SA-LSTM Learner')

# Data preparation
def load_and_prepare_data(url: str, max_features: int, maxlen: int, test_dim: float) -> Tuple:
    data = pd.read_csv(url)
    return util.prepare_data(data, max_features, maxlen, test_dim)

X_train, X_val, X_test, Y_train, Y_val, Y_test, word_index, tokenizer = load_and_prepare_data(
    URL_MODEL_DATA_PATH, MAX_FEATURES, MAXLEN, TEST_DIM
)

# Model creation
def build_model(input_shape: Tuple[int], embed_dim: int, max_features: int) -> Model:
    with strategy.scope():
        model_input = Input(shape=input_shape, name="input")
        x = Embedding(max_features, embed_dim, input_length=input_shape[0], name="embedding")(model_input)
        x = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(x)
        x = Dense(128, activation='relu', kernel_initializer='uniform')(x)
        x = Dense(64, activation='relu', kernel_initializer='uniform')(x)
        model_output = Dense(2, activation='sigmoid', kernel_initializer='uniform')(x)

        model = Model(inputs=model_input, outputs=model_output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

model = build_model((MAXLEN,), EMBED_DIM, MAX_FEATURES)
model.summary()

# Training function
def train_model(model: Model, X_train, Y_train, X_val, Y_val, batch_size: int, epochs: int) -> dict:
    logger.info('Starting training...')
    start = datetime.datetime.now()
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_val, Y_val), verbose=1)
    duration = (datetime.datetime.now() - start).total_seconds() * 1000
    logger.info(f'Training completed in {duration:.2f}ms')
    return history.history

history = train_model(model, X_train, Y_train, X_val, Y_val, BATCH_SIZE, EPOCHS)

# Save model weights
def save_model(model: Model, path: str) -> None:
    try:
        model.save(path)
        logger.info(f'Model weights saved to {path}')
    except Exception as e:
        logger.error(f'Failed to save model weights: {e}')

save_model(model, MODEL_FILE_PATH)

# Evaluation function
def evaluate_model(model: Model, X, Y, batch_size: int, dataset_name: str) -> None:
    logger.info(f'Evaluating {dataset_name} dataset...')
    start = datetime.datetime.now()
    scores = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
    duration = (datetime.datetime.now() - start).total_seconds() * 1000

    logger.info(f'{dataset_name} evaluation completed in {duration:.2f}ms')
    for metric, score in zip(model.metrics_names, scores):
        logger.info(f'{dataset_name} - {metric}: {score:.4f}')

evaluate_model(model, X_train, Y_train, BATCH_SIZE, 'Train')
evaluate_model(model, X_test, Y_test, BATCH_SIZE, 'Test')

# Metrics computation
def compute_metrics(Y_true, Y_pred) -> None:
    f1 = f1_score(Y_true, Y_pred, average="binary")
    recall = recall_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred)
    conf_matrix = confusion_matrix(Y_true, Y_pred)

    logger.info(f'Test - F1 Score: {f1:.4f}')
    logger.info(f'Test - Recall: {recall:.4f}')
    logger.info(f'Test - Precision: {precision:.4f}')
    logger.info(f'Test - Confusion Matrix:\n{conf_matrix}')

Y_test_binary = np.argmax(Y_test, axis=1)
Y_pred_binary = np.argmax(model.predict(X_test, batch_size=1024), axis=1)

compute_metrics(Y_test_binary, Y_pred_binary)
