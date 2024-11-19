#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path
import datetime
import sys
import signal
import logging
import logging.handlers
import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense

class SampleTWINServerBandit:
    MODEL_TRAINING_EPOCHS = 50
    # BATCH_SIZE = 32
    URL_MODEL_DATA_PATH = 'https://www.inf.ufrgs.br/~cakunas/data/access-pattern-metrics-train.csv'
    # MODEL_DATA_PATH = 'access-pattern-metrics-train.csv'
    MODEL_FILE_PATH = 'tf-model-keras-trained-{}.keras'.format(datetime.datetime.now().strftime('%Y-%m-%d'))
    LOG_FILENAME = 'sample.log'
    DATABASE = {}

    # Construtor
    def __init__(self, debug):

        self.configureLog(debug)
        self.configureSignal()

    # Signal
    def configureSignal(self):
        signal.signal(signal.SIGINT, self.signalHandler)

    # Handle the signals we capture
    def signalHandler(self, sig, frame):
        self.logger.info('Shutdown requested (SIGINT)')

        # Display the statistics of the server
        self.logger.info('N = {}'.format(self.N))
        self.logger.info('Q = {}'.format(self.Q))

        sys.exit(0)

    # Configure the log system
    def configureLog(self, debug):
        # Creates and configure the log file
        self.logger = logging.getLogger('TWINServerBandit')

        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # Defines the format of the logger
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # Configure the log rotation
        handler = logging.handlers.RotatingFileHandler(self.LOG_FILENAME, maxBytes=268435456, backupCount=50, encoding='utf8')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.logger.info('Starting TWINS Council')

    def trainModel(self):
        # Detect hardware
        try:
            tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
        except ValueError:
            tpu_resolver = None
            gpus = tf.config.experimental.list_logical_devices("GPU")

        # Select appropriate distribution strategy
        if tpu_resolver:
            tf.config.experimental_connect_to_cluster(tpu_resolver)
            tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
            strategy = tf.distribute.TPUStrategy(tpu_resolver)
            print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
        elif len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
            print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
        elif len(gpus) == 1:
            strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
            print('Running on single GPU ', gpus[0].name)
        else:
            strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
            print('Running on CPU')
            
        print("Number of accelerators: ", strategy.num_replicas_in_sync)

        BATCH_SIZE = 32 * strategy.num_replicas_in_sync # Gobal batch size.
        # The global batch size will be automatically sharded across all
        # replicas by the tf.data.Dataset API. A single TPU has 8 cores.
        # The best practice is to scale the batch size by the number of
        # replicas (cores). The learning rate should be increased as well.

        csv_path = pd.read_csv(self.URL_MODEL_DATA_PATH, index_col=0, sep=',')

        header = csv_path.dtypes
        self.logger.debug('Features: {}'.format(header))

        df = csv_path.values
        self.logger.debug('Dataframe: {}'.format(df.shape))

        x = df[:, 0:df.shape[1] - 1]    # 0 ... < N-1
        y = df[:, df.shape[1] - 1]      # N-1

        # Apply data transformations
        self.logger.debug('TensorFlow - Applying data transformations')

        self.transformPower = PowerTransformer(method='yeo-johnson').fit(x)
        self.logger.debug('TensorFlow - YeoJohnson - lambdas: {}'.format(self.transformPower.lambdas_))

        # Make the transformation
        x = self.transformPower.transform(x)

        self.transformScaler = StandardScaler().fit(x)
        self.logger.debug('TensorFlow - Scaling - mean: {}'.format(self.transformScaler.mean_))

        # Make the transformation
        x = self.transformScaler.transform(x)

        self.logger.debug('TensorFlow - Binarizer:')

        self.transformBinarizer = LabelBinarizer()
        self.transformBinarizer.fit_transform(y)

        y = self.transformBinarizer.transform(y)

        self.logger.info('TensorFlow - Features: {}'.format(x.shape))
        self.logger.info('TensorFlow - Output: {}'.format(y.shape))

        # Split the dataset for training and testing
        self.logger.info('TensorFlow - Splitting the dataset for training and testing')

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=0.3
        )

        # Check if a saved model exists
        if os.path.isfile(self.MODEL_FILE_PATH):
            # Load the model
            self.logger.info('TensorFlow - Loading the Neural Network model')

            self.model = load_model(self.MODEL_FILE_PATH)
        else:
            with strategy.scope():
                # Create a new model and train it
                self.logger.info('TensorFlow - Creating the Neural Network model')
                self.model = Sequential()
                self.model.add(Dense(units=x_train.shape[1], input_dim=x_train.shape[1], activation='relu', kernel_initializer='normal', name='input_agios_metrics'))
                self.model.add(Dense(units=x_train.shape[1], activation='relu', kernel_initializer='normal', name='hidden_layer'))
                self.model.add(Dense(units=3, activation='softmax', name='output_layer'))

                self.model.compile(
                    optimizer=RMSprop(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                self.logger.debug('TensorFlow - Summary: {}'.format(self.model.summary()))

            self.logger.info('TensorFlow - Start training')


            start = datetime.datetime.now()
 
            self.model.fit(
                x_train, y_train,
                epochs=self.MODEL_TRAINING_EPOCHS,
                verbose=1,
                batch_size=BATCH_SIZE, #32
                validation_split=0.2
            )

            end = datetime.datetime.now()

            diff = end - start
            self.logger.info('TensorFlow - TRAIN - Time: {}ms'.format(diff.total_seconds() * 1000))

            # Save the model
            self.model.save(self.MODEL_FILE_PATH)

            self.logger.info('TensorFlow - TRAIN - Evaluating dataset')

            start = datetime.datetime.now()
            score = self.model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
            end = datetime.datetime.now()

            diff = end - start
            self.logger.info('TensorFlow - TRAIN - Time: {}ms'.format(diff.total_seconds() * 1000))

            for i in range(0, len(self.model.metrics_names)):
                self.logger.info('TensorFlow - TRAIN - {}: {}'.format(self.model.metrics_names[i], score[i]))

            score = self.model.predict(x_test, batch_size=1)

            self.logger.info('TensorFlow - TEST - R2: {}'.format(r2_score(y_test, score)))

            self.logger.info('TensorFlow - TEST - Evaluating dataset')

            start = datetime.datetime.now()
            score = self.model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
            end = datetime.datetime.now()

            diff = end - start
            self.logger.info('TensorFlow - TEST - Time: {}ms'.format(diff.total_seconds() * 1000))

            for i in range(0, len(self.model.metrics_names)):
                self.logger.info('TensorFlow - TEST - {}: {}'.format(self.model.metrics_names[i], score[i]))

            score = self.model.predict(x_test, batch_size=1)

            self.logger.info('TensorFlow - TEST - R2: {}'.format(r2_score(y_test, score)))
