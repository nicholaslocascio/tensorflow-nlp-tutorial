import csv
import pandas as pd
import re
import special_tokens
import np
import random
import tensorflow as tf

def load_arithmetic_data():
    data = pd.read_csv("./data/arithmetic-data.csv")
    equations = list(data['input'])
    answers = list(data['output'])
    print(answers[0])

    equations_of_chars = [[c for c in str(equation)] for equation in equations]
    answers_of_chars = [[c for c in str(answer)] for answer in answers]
    index_to_char = {i: str(i) for i in range(1, 10)}
    index_to_char[len(index_to_char.keys())+1] = '+'
    index_to_char[len(index_to_char.keys())+1] = '*'
    index_to_char[len(index_to_char.keys())+1] = '-'
    vocab_size = len(index_to_char.keys())
    char_to_index = {v:k for k,v in index_to_char.iteritems()}

    max_len = max(np.max([len(equation_of_chars) for equation_of_chars in equations_of_chars]), np.max([len(answer_of_chars) for answer_of_chars in answers_of_chars]))
    n_equations = len(equations_of_chars)

    X = np.zeros(shape=(n_equations, max_len, vocab_size), dtype='float32')
    y = np.zeros(shape=(n_equations, max_len, vocab_size), dtype='float32')

    label_to_index = {'negative': 0, 'neutral': 1, 'positive':2}
    for sentence_index in range(n_equations):
        current_sentence = equations_of_chars[sentence_index]
        for current_char_position in range(len(current_sentence)):
            char = equations_of_chars[sentence_index][current_char_position]
            token_index = char_to_index[char]
            X[sentence_index][current_char_position][token_index] = 1

        current_answer = answers_of_chars[sentence_index]
        for current_char_position in range(len(current_answer)):
            char = answers_of_chars[sentence_index][current_char_position]
            token_index = char_to_index[char]
            y[sentence_index][current_char_position][token_index] = 1
    return X, y, index_to_char, equations, answers, max_len


def split_data(X, y, train_split=0.8, dev_split=0.1, test_split=0.1, random=False):
    """Splits data"""
    num_examples = len(X)
    indices = range(X.shape[0])
    if random:
        random.seed(42)
        random.shuffle(indices)
    boundary = int(num_examples*train_split)
    training_idx, test_idx = indices[:boundary], indices[boundary:]
    X_train, X_test = X[training_idx,:], X[test_idx,:]
    y_train, y_test = y[training_idx,:], y[test_idx,:]

    return X_train, y_train, X_test, y_test
