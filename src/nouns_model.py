# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
import csv
from json import loads
import tensorflow as tf
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

def main():
    labels: list[int]
    nouns: list
    INPUT_LENGTH=50

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    ## open file and load data
    with open('data/nouns.csv', 'r') as nouns_csv:
        data = list(csv.reader(nouns_csv))
        labels = [int(row[0]) for row in data]
        nouns = [row[1] for row in data]
        pass

    ## Tokenise and prepare word data
    tokenizer = Tokenizer(num_words=50, char_level=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(nouns)
    noun_data = tokenizer.texts_to_sequences(nouns)
    noun_data = pad_sequences(noun_data, padding='post', maxlen=INPUT_LENGTH)

    ## Create NN
    model = keras.models.Sequential()
    model.add(keras.layers.Input(INPUT_LENGTH,))
    model.add(keras.layers.Dense(128, activation='relu', name="First_Layer"))
    model.add(keras.layers.Dense(1, activation='sigmoid', name="Output_Layer"))
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train NN
    labels_array = array(labels, dtype='int')
    model.fit(noun_data, labels_array, epochs=50)

    ## Predict a word
    test_nouns = ['سَاخِرُون', 'سَاخِر']
    test_nouns_tok = tokenizer.texts_to_sequences(test_nouns)
    test_nouns_tok = pad_sequences(test_nouns_tok, padding='post', maxlen=INPUT_LENGTH)

    print(model.predict(test_nouns_tok))



if __name__ == '__main__':
    main()
