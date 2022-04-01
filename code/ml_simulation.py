import numpy as np
import tensorflow as tf
from tensorflow import keras
import data_analysis_functions as da
import data_process

class ML_Model:
    def __init__(self, gene):
        self.gene = gene

    def convert_to_model(self):
        model = keras.Sequential()
        model.add(keras.Input(shape=(300,)))
        model.add(keras.layers.Embedding(30523, 256))
        model.add(keras.layers.Conv1D(filters = 150, kernel_size = 2))
        for gen in self.gene:
            if gen[0] == 0:
                model.add(keras.layers.Conv1D(filters = gen[2], kernel_size = gen[1]))
            elif gen[0] == 1:
                model.add(keras.layers.Dense(gen[1], activation = 'tanh'))
            elif gen[0] == 2:
                model.add(keras.layers.MaxPooling1D(gen[1]))
            elif gen[0] == 3:
                model.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(gen[1], return_sequences=True), input_shape=(5, 10)))
                model.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(gen[1])))
        model.add(keras.layers.Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

class ML_simulation:

    def __init__(self, tokenizer, model, train_df, test_df):
        self.tokenizer = tokenizer
        self.model = model
        self.train_df = 'data/training_set.csv'
        self.test_df = 'data/test_set.csv'

    @staticmethod
    def filter_long_descriptions(tokenizer, descriptions, max_len):
        indices = []
        lengths = tokenizer(descriptions, padding=False,
                         truncation=False, return_length=True)['length']
        for i in range(len(descriptions)):
            if lengths[i] <= max_len - 2:
                indices.append(i)
        return indices

    def short_description(self, df):
        # print(df.cleaned_comment.tolist())
        short_descriptions = self.filter_long_descriptions(self.tokenizer, df.cleaned_comment.tolist(), 300)
        # print('short descriptions')
        # print(short_descriptions)
        data = np.array(df['encoded_comment'])[short_descriptions]
        # print(data)
        target = df['score']
        return data, target

    def simulation(self):
        train_df = da.importData(self.train_df)
        test_df = da.importData(self.test_df)

        train_df = data_process.data_vectorization(train_df)
        test_df = data_process.data_vectorization(test_df)

        train_data, train_target = self.short_description(train_df)
        test_data, test_target = self.short_description(test_df)

        # print(train_data)
        for i in range(len(train_data)):
            train_data[i] = np.array(train_data[i]).astype(np.float32)
            train_data[i] = train_data[i].flatten()

        self.model.fit(np.stack(train_data, 0), train_target, epochs = 7)
        result = self.model.evaluate(np.stack(test_data, 0), test_target)

        print(result)
        if result < 0.045:
            print('possilbe good model')
            print(self.model.summary())

        keras.backend.clear_session()
        return result
