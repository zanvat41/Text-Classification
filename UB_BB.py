import sys
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np


def load_files_body(dir):
    rawData = load_files(dir)

    # remove the header
    for i in range(len(rawData.data)):
        ll = rawData.data[i].decode("utf-8", "ignore")
        lines = ll.splitlines()
        for j, line in enumerate(lines):
            # find the gap line between header and body
            if ':' not in line:
                rawData.data[i] = str.encode('\n'.join(lines[j + 1:]))
                break

    return rawData


def generate_output(test, pred):
    precision = metrics.precision_score(test.target, pred, average='macro')
    recall = metrics.recall_score(test.target, pred, average='macro')
    f1_score = metrics.f1_score(test.target, pred, average='macro')
    return str(precision) + "," + str(recall) + "," + str(f1_score)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        train = load_files_body(sys.argv[1])
        test = load_files_body(sys.argv[2])
        out_file = open(sys.argv[3], 'w')
        display = sys.argv[4]
    else:
        print('Wrong number of arguments. Usage:\npython UB_BB.py trainset testset output display_LC')
        sys.exit()

    res = {}
    for i in range(2):
        # Unigram
        if i == 0:
            count_vect = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
            train_data = count_vect.fit_transform(train.data)
            ub_train_data = train_data
            test_data = count_vect.transform(test.data)
            ub_test_data = test_data
            ub_bb = 'UB'
        # Bigram
        else:
            count_vect = CountVectorizer(ngram_range=(1, 2), decode_error='ignore')
            train_data = count_vect.fit_transform(train.data)
            test_data = count_vect.transform(test.data)
            ub_bb = 'BB'

        # Naive Bayes
        algo_config = 'NB' + ub_bb
        clf = MultinomialNB()
        clf.fit(train_data, train.target)
        pred = clf.predict(test_data)
        res[algo_config] = pred

        # Logistic Regression
        algo_config = 'LR' + ub_bb
        clf = LogisticRegression()
        clf.fit(train_data, train.target)
        pred = clf.predict(test_data)
        res[algo_config] = pred

        # Support Vector Machines
        algo_config = 'SVM' + ub_bb
        clf = SVC(kernel='linear')
        clf.fit(train_data, train.target)
        pred = clf.predict(test_data)
        res[algo_config] = pred

        # Random Forest
        algo_config = 'RF' + ub_bb
        clf = RandomForestClassifier()
        clf.fit(train_data, train.target)
        pred = clf.predict(test_data)
        res[algo_config] = pred

    # Print result to output file in the order specified
    out_file.write("NB,UB,"+generate_output(test, res['NBUB'])+"\n")
    out_file.write("NB,BB,"+generate_output(test, res['NBBB'])+"\n")
    out_file.write("LR,UB,"+generate_output(test, res['LRUB'])+"\n")
    out_file.write("LR,BB,"+generate_output(test, res['LRBB'])+"\n")
    out_file.write("SVM,UB,"+generate_output(test, res['SVMUB'])+"\n")
    out_file.write("SVM,BB,"+generate_output(test, res['SVMBB'])+"\n")
    out_file.write("RF,UB,"+generate_output(test, res['RFUB'])+"\n")
    out_file.write("RF,BB,"+generate_output(test, res['RFBB'])+"\n")
    out_file.close()

    # Show learning curve if necessary
    if display == '1':
        train_data = ub_train_data
        test_data = ub_test_data
        step_size = 200.0
        train_size = train_data.shape[0]
        data = {}
        for i in range(int(np.ceil(train_size / step_size))):
            current_size = (i + 1) * step_size
            if current_size < train_size:
                current_size = int(current_size)
            else:
                current_size = int(train_size)

            # Naive Bayes
            algo_config = 'NB UB'
            clf = MultinomialNB()
            clf.fit(train_data[:current_size], train.target[:current_size])
            pred = clf.predict(test_data)
            f1_score = metrics.f1_score(test.target, pred, average='macro')
            data.setdefault(algo_config, {'sizes': [], 'f1_scores': []})
            data[algo_config]['sizes'].append(current_size)
            data[algo_config]['f1_scores'].append(f1_score)

            # Logistic Regression
            algo_config = 'LR UB'
            clf = LogisticRegression()
            clf.fit(train_data[:current_size], train.target[:current_size])
            pred = clf.predict(test_data)
            f1_score = metrics.f1_score(test.target, pred, average='macro')
            data.setdefault(algo_config, {'sizes': [], 'f1_scores': []})
            data[algo_config]['sizes'].append(current_size)
            data[algo_config]['f1_scores'].append(f1_score)

            # Support Vector Machines
            algo_config = 'SVM UB'
            clf = SVC(kernel='linear')
            clf.fit(train_data[:current_size], train.target[:current_size])
            pred = clf.predict(test_data)
            f1_score = metrics.f1_score(test.target, pred, average='macro')
            data.setdefault(algo_config, {'sizes': [], 'f1_scores': []})
            data[algo_config]['sizes'].append(current_size)
            data[algo_config]['f1_scores'].append(f1_score)

            # Random Forest
            algo_config = 'RF UB'
            clf = RandomForestClassifier()
            clf.fit(train_data[:current_size], train.target[:current_size])
            pred = clf.predict(test_data)
            f1_score = metrics.f1_score(test.target, pred, average='macro')
            data.setdefault(algo_config, {'sizes': [], 'f1_scores': []})
            data[algo_config]['sizes'].append(current_size)
            data[algo_config]['f1_scores'].append(f1_score)

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.title('Learning curves')
        plt.ylim(0.4, 0.95)
        plt.xlabel("Size of training data")
        plt.ylabel("F1 Score")
        plt.grid()
        clrs = ['red', 'blue', 'green', 'yellow']
        for algo_config, clr in zip(data, clrs):
            sizes, f1_scores = data[algo_config]['sizes'], data[algo_config]['f1_scores']
            color = colors.cnames[clr]
            plt.plot(sizes, f1_scores, 'o-', color=color, label=algo_config)
        plt.legend(loc=(0.7, -.38), prop=dict(size=14))
        plt.show()