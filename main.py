from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import callbacks
from csv_parser import CsvParser
from data_labels import determine_labels, plot
import tensorflow as tf

if __name__ == '__main__':
    psr = CsvParser('../saham/')
    _data = psr.get_one('شتران', 30)
    _closes = [float(x["fields"]["close"]) for x in _data]
    _opens = np.array([float(x["fields"]["open"]) for x in _data]).reshape([-1, 1])
    _highs = np.array([float(x["fields"]["high"]) for x in _data]).reshape([-1, 1])
    _lows = np.array([float(x["fields"]["low"]) for x in _data]).reshape([-1, 1])
    _volumes = np.array([float(x["fields"]["volume"]) for x in _data]).reshape([-1, 1])
    print("data fetch successfully")
    _labels = determine_labels(_closes)
    plot(_closes, _labels)
    _closes = np.array(_closes).reshape([-1, 1])
    _raw_data = np.concatenate([_closes, _opens, _highs, _lows, _volumes], axis=1)
    _data = _raw_data
    pre = 25
    for i in range(pre):
        _raw_data = np.concatenate([np.array([0] * _raw_data.shape[1]).reshape([1, -1]), _raw_data[:-1]], axis=0)
        _data = np.concatenate([_raw_data, _data], axis=1)
    _data = _data[pre:]
    _labels = _labels[pre:]
    _labels = np.array(_labels)
    # perm = tf.random.shuffle(tf.range(tf.shape(_data)[0]))
    # _data = tf.gather(_data, perm, axis=0)
    # _labels = tf.gather(_labels, perm, axis=0)
    test_index = int(0.95 * len(_data))
    _train_data, _validation_data = _data[:test_index], _data[test_index:]
    _train_labels, _validation_labels = _labels[:test_index], _labels[test_index:]
    print("getting data from saham file completed")
    my_minmax = MinMaxScaler()
    my_minmax.fit(_train_data)
    print("min max ")
    _train_data = my_minmax.transform(_train_data)
    _validation_data = my_minmax.transform(_validation_data)
    # model
    size = _data.shape[1]
    input_layer = Input((size,))
    hidden_layer = Dense(units=4, activation='relu')(input_layer)
    hidden_layer = Dense(units=8, activation='relu')(hidden_layer)
    hidden_layer = Dense(units=16, activation='relu')(hidden_layer)
    hidden_layer = Dense(units=8, activation='relu')(hidden_layer)
    hidden_layer = Dense(units=4, activation=None)(hidden_layer)
    prediction = Dense(units=1)(hidden_layer)
    early_stop = callbacks.EarlyStopping('loss', patience=5)
    my_first_regression_model = Model(inputs=input_layer, outputs=prediction)
    my_first_regression_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')
    print("validation")
    my_first_regression_model.evaluate(_validation_data, _validation_labels)
    my_first_regression_model.evaluate(_train_data, _train_labels)
    print("train starting...")
    my_first_regression_model.fit(x=_train_data, y=_train_labels, batch_size=200,
                                  validation_split=0.1,
                                  callbacks=[early_stop],
                                  epochs=250)
    print("validation")
    my_first_regression_model.evaluate(_validation_data, _validation_labels)
    my_first_regression_model.evaluate(_train_data, _train_labels)
