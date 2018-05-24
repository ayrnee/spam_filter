#!/usr/bin/env python
from collections import defaultdict
import numpy as np
import time

def build_vocab(data):
    word_count = defaultdict(lambda : 0)
    for email in data:
        temp_set = set()
        for word in email:
            temp_set.add(word)
        for word in temp_set:
            word_count[word] += 1
    # vocab = {key: count for key, count in word_count.iteritems() if count > 30}
    vocab = [key for key in word_count.keys() if word_count[key] > 30]
    # print vocab
    return vocab

def partition_data(data, split_on):
    return data[:split_on], data[split_on:]

def clean_data(raw_data):
    label = []
    data = []
    for i in range(len(raw_data)):
        temp = -1 if int(raw_data[i][:1]) == 0 else 1
        label.append(temp)
        data.append(raw_data[i][1:].strip().split())
    # print data
    # print label
    return label, data

def read_data(f_name):
    with open(f_name) as file:
        return file.readlines()

def feature_transform(item, vocab):
    sample = [0] * len(vocab)
    for i in range(len(vocab)):
        sample[i] = 1 if vocab[i] in item else 0
    return np.array(sample)

def build_feature_list(data, vocab):
    samples = [feature_transform(item, vocab) for item in data]
    return samples

def is_misclassified(w, data):
    train_x = data["train_x"]
    train_y = data["train_y"]

    for i in range(len(train_x)):
        pred = np.dot(train_x[i],w)
        y = 1 if pred >= 0 else -1
        if train_y[i] !=  y:
            return True

    return False

def perceptron_train_avg(data, max_itr = 100):
    train_x = data["train_x"]
    train_y = data["train_y"]
    w_vecs = []
    w = np.zeros(len(train_x[0]))
    k = 0
    itr = 0
    while (is_misclassified(w, data) and itr < max_itr):
        for i in range(len(train_x)):
            # print train_x[i]
            pred = np.dot(train_x[i],w)
            y = 1 if pred >= 0 else -1
            # if (train_y[i] * y) < 0:
            if y != train_y[i]:
                k = k + 1
                w = np.add(w, train_y[i] * train_x[i])
            w_vecs.append(w)
        itr = itr + 1
    w_vecs = np.mean(np.array(w_vecs), axis=0)
    # print w_vecs

    return w_vecs, k, itr

def perceptron_train(data, max_itr = 100):
    train_x = data["train_x"]
    train_y = data["train_y"]

    w = np.zeros(len(train_x[0]))
    k = 0
    itr = 0
    while (is_misclassified(w, data) and itr < max_itr):
        for i in range(len(train_x)):
            # print train_x[i]
            pred = np.dot(train_x[i],w)
            y = 1 if pred >= 0 else -1
            # if (train_y[i] * y) < 0:
            if y != train_y[i]:
                k = k + 1
                w = np.add(w, train_y[i] * train_x[i])
        itr = itr + 1
    return w, k, itr

def perceptron_test(w, data, x , y):
    data_x = data[x]
    data_y = data[y]

    k = 0
    for i in range(len(data_x)):
        pred = np.dot(data_x[i], w)
        y = 1 if pred >= 0 else -1
        if data_y[i] != y:
            k = k + 1

    return (k * 1. / len(data_x))

def relevant_weights(w, vocab, count):
    sorted_weights = np.argsort(w)
    most_impact = []
    least_impact = []
    for i in range(count):
        most_impact.append(vocab[sorted_weights[i]])
        least_impact.append(vocab[sorted_weights[len(sorted_weights) - i - 1]])
    return most_impact, least_impact

def training_data_impact(raw_data):
    steps = [100, 200, 400, 800, 2000, 4000]
    for step in steps:
        print "Using N = " + str(step)
        training, validation = partition_data(raw_data, step)
        training = clean_data(training)
        validation = clean_data(validation)
        vocab = build_vocab(training[1])
        samples = build_feature_list(training[1], vocab)
        data = {}
        data["train_x"] = samples
        data["train_y"] = training[0]
        w, k, itr = perceptron_train_avg(data, 1000)
        print "The error on the training set using the averaged perceptron is: " + str(perceptron_test(w, data, "train_x", "train_y"))
        validation_samples = build_feature_list(validation[1], vocab)
        data["val_x"] = validation_samples
        data["val_y"] = validation[0]
        print "The error on the validation set using the averaged perceptron is: " + str(perceptron_test(w, data, "val_x", "val_y"))

        w, k, itr = perceptron_train(data, 1000)
        print "The error on the training set using the perceptron is: " + str(perceptron_test(w, data, "train_x", "train_y"))
        validation_samples = build_feature_list(validation[1], vocab)
        data["val_x"] = validation_samples
        data["val_y"] = validation[0]
        print "The error on the validation set using the perceptron is: " + str(perceptron_test(w, data, "val_x", "val_y"))


def main():
    tic = time.time()
    raw_data = read_data("./data/spam_train.txt")
    print "File opened"
    training_data_impact(raw_data)
    toc = time.time()
    print toc-tic
    # training, validation = partition_data(raw_data, 4000)
    # training = clean_data(training)
    # validation = clean_data(validation)
    # print "Data pre-processed"
    # vocab = build_vocab(training[1])
    # print "Vocabulary built"
    # samples = build_feature_list(training[1], vocab)
    # print "Feature matrix formed"
    # data = {}
    # data["train_x"] = samples
    # data["train_y"] = training[0]
    # w, k, itr = perceptron_train_avg(data, 1000)
    # # print w
    # print "The number of misclasified samples: "
    # print k
    # # print itr
    # print "The error on the training set is: " + str(perceptron_test(w, data, "train_x", "train_y"))
    # validation_samples = build_feature_list(validation[1], vocab)
    # print "Validation samples matrix formed"
    # data["val_x"] = validation_samples
    # data["val_y"] = validation[0]
    # print "The accuracy on the validation set is: " + str(1 - perceptron_test(w, data, "val_x", "val_y"))

    # test_data = read_data("./data/spam_test.txt")
    # test_data = clean_data(test_data)
    # test_samples = build_feature_list(test_data, vocab)
    # data["test_x"] = test_samples
    # data["test_y"] = test_data[0]
    # # print "The accuracy on test data is: " + str(1 - perceptron_test(w, data, "test_x", "test_y"))
    # most_impact, least_impact = relevant_weights(w, vocab, 15)
    # # print most_impact
    # # print least_impact
if __name__ == "__main__":
    main()
