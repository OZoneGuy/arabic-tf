### NOTE: Bad idea!!!

from csv import reader
# from json import loads
import tensorflow as tf
from numpy import Infinity, array, matrix
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

LETTER_START: int = int('0621', 16)
LETTER_END:   int = int('064A', 16)

ACCENT_START: int = int('064B', 16)
ACCENT_END:   int = int('0652', 16)

START_CODE: int = int('0621', 16)
END_CODE:   int = int('0652', 16)
CHARACTER_RANGE = END_CODE - START_CODE

MAX_LENGTH = 20

def get_data(file_path: str)-> dict[str, list[str]]:
    data: list[list[str]]
    with open(file_path, 'r') as csv_file:
        raw = list(reader(csv_file))
        data = array(raw).T.tolist()
        pass

    data_dict: dict[str, list[str]]  = {}
    for column in data:
        data_dict[column[0]] = column[1:]
        pass
    return data_dict


def get_model():
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=(MAX_LENGTH, CHARACTER_RANGE)))
    # model.add(keras.layers.Dense(5000, activation='sigmoid'))
    model.add(keras.layers.Dense(MAX_LENGTH*CHARACTER_RANGE, activation='softmax'))
    return model


def word_to_matrix(word: str) -> matrix:
    def letter_to_index(letter: str) -> int:
        index = ord(letter) - START_CODE
        return index

    matrix_list = [[0 for _ in range(CHARACTER_RANGE)] for _ in range(MAX_LENGTH)]
    for (i, c) in enumerate(word):
        matrix_list[i][letter_to_index(c)] = 1
    return matrix(matrix_list)


def matrix_to_word(matrix: matrix)-> str:
    l: list[list[float]] = matrix.tolist()
    def vector_to_letter(v: list[float])-> str:
        index = 0
        cur_max = -Infinity
        for (i, n) in enumerate(v):
            if n > cur_max:
                index = i
                cur_max = n
                pass
            pass
        return chr(index+START_CODE)

    return ''.join(map(lambda v: vector_to_letter(v), l))


def get_character_type(c: str)-> int:
    if (int(c) <= LETTER_END) and (int(c) >= LETTER_START):
        return 0
    elif (int(c) == int('0651', 16)):
        return 2
    elif (int(c) <= ACCENT_END) and (int(c) >= ACCENT_START):
        return 1
    else:
        return -1


def char_to_int(c: str)-> int:
    c_type = get_character_type(c)
    if c_type == 0:
        return int(c) - LETTER_START
    elif c_type == 1:
        return int(c) - ACCENT_START
    return -1

def word_to_vectors(word: str)-> matrix:
    l: list[list[int]] = []
    for (n, c) in enumerate(word):
        c_type = get_character_type(c)
        if c_type == 0:
            l.append([0, char_to_int(c)])
        elif c_type == 1:
            l[-1][0] = char_to_int(c)
        elif c_type == 2:
            l[-1][0] == 7
            l.append(l[-1].copy())
        else:
            raise Exception(f'Bad character: {c} at  position {n}')
        pass
    l += [[0,0]] * (MAX_LENGTH - len(l))
    print(l)
    return matrix(l)

def setup_gpu():
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


def main():
    # setup_gpu()

    file_path = 'data/verbs.csv'
    ## Data dictionary configuration
    # root(m,s) <--- Used as the source/input, all else will be used to train different NN
    # root(m,p)
    # root(f,s)
    # root(f,p)
    # present(m,s)
    # present(m,p)
    # present(f,s)
    # present(f,p)
    # command(m,s)
    # command(m,p)
    # command(f,s)
    # command(f,p)
    data = get_data(file_path)

    input_raw = data['root(m,s)']

    output_raw = data['root(m,p)']

    pass


if __name__ == '__main__':
    main()
