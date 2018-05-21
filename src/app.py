from collections import defaultdict
import numpy as np

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

def preprocess(raw_data):
    train_set, val_set = partition_data(raw_data, 4000)
    train_set = clean_data(train_set)
    val_set = clean_data(val_set)
    return train_set, val_set

def feature_transform(item, vocab):
    sample = [0] * len(vocab)
    for i in range(len(vocab)):
        sample[i] = 1 if vocab[i] in item else 0
    return np.array(sample)

def build_feature_list(data, vocab):
    samples = [feature_transform(item, vocab) for item in data]
    return samples

def perceptron_train(data):
    train_x = data["train_x"]
    train_y = data["train_y"]

    w = np.zeros(len(train_x[0]))
    k = 0
    itr = 100
    for t in range(itr):
        for i in range(len(train_x)):
            # print train_x[i]
            pred = np.dot(train_x[i],w)
            y = 1 if pred >= 0 else -1
            if (train_y[i] * y) < 0:
                k = k + 1
                w = np.add(w, train_y[i] * train_x[i])
    return w, k, itr

def main():
    raw_data = read_data("./data/spam_train.txt")
    print "File opened"
    training, validation = preprocess(raw_data)
    print "Data pre-processed"
    vocab = build_vocab(training[1])
    print "Vocabulary built"
    samples = build_feature_list(training[1], vocab)
    print "Feature matrix formed"
    data = {}
    data["train_x"] = samples
    data["train_y"] = training[0]
    w, k, itr = perceptron_train(data)
    print w
    print k
    print itr

if __name__ == "__main__":
    main()
