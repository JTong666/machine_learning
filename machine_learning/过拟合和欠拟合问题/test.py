import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt



NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results


train_data = multi_hot_sequences(train_data, NUM_WORDS)
test_data = multi_hot_sequences(test_labels, NUM_WORDS)


baseline_model = keras.Sequential()
baseline_model.add(keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS, )))
baseline_model.add(keras.layers.Dense(16, activation=tf.nn.relu))
baseline_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

baseline_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)




smaller_model = keras.Sequential()
smaller_model.add(keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS, )))
smaller_model.add(keras.layers.Dense(4, activation=tf.nn.relu))
smaller_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


smaller_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])


smaller_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)




def smaller_model():#正则化和随机丢弃
    s_model = keras.Sequential()
    s_model.add(keras.layers.Dense(4, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(NUM_WORDS, )))
    s_model.add(keras.layers.Dropout(0.5))
    s_model.add(keras.layers.Dense(4, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu))
    s_model.add(keras.layers.Dropout(0.5))
    s_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    s_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])


    return s_model



