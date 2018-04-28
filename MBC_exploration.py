import sys
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def load_files_body(dir, stem=True, stop=True):
    rawData = load_files(dir)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    for i in range(len(rawData.data)):
        ll = rawData.data[i].decode("utf-8", "ignore")
        lines = ll.splitlines()
        new_lines = []
        is_header = True
        for j, line in enumerate(lines):
            if is_header and ':' in line:
                continue
            is_header = False
            if (stem or stop) and line:
                words = line.split(' ')
                new_words = []
                for word in words:
                    if stop and word.lower() in stop_words:
                        continue
                    if stop:
                        word = word.lower()
                    if stem:
                        word = str(stemmer.stem(word))
                    new_words.append(word)
                new_line = ' '.join(new_words)
            else:
                new_line = line
            new_lines.append(new_line)
        rawData.data[i] = str.encode('\n'.join(new_lines))
    return rawData


if __name__ == '__main__':
    if len(sys.argv) == 4:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        out_file = open(sys.argv[3], 'w')
    else:
        print('Wrong number of arguments. Usage:\npython MBC_exploration.py trainset testset output')
        sys.exit()

    # Load files and get data
    train_none_none = load_files_body(train_path, stem=False, stop=False)
    test_none_none = load_files_body(test_path, stem=False, stop=False)
    train_stem_stop = load_files_body(train_path, stem=True, stop=True)
    test_stem_stop = load_files_body(test_path, stem=True, stop=True)
    train_none_stop = load_files_body(train_path, stem=False, stop=True)
    test_none_stop = load_files_body(test_path, stem=False, stop=True)
    train_stem_none = load_files_body(train_path, stem=True, stop=False)
    test_stem_none = load_files_body(test_path, stem=True, stop=False)

    # Vectorize data
    tfid_none_none = TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')
    tfid_train_none_none = tfid_none_none.fit_transform(train_none_none.data)
    tfid_test_none_none = tfid_none_none.transform(test_none_none.data)

    tfid_stem_stop = TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')
    tfid_train_stem_stop = tfid_stem_stop.fit_transform(train_stem_stop.data)
    tfid_test_stem_stop = tfid_stem_stop.transform(test_stem_stop.data)

    tfid_stem_none = TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')
    tfid_train_stem_none = tfid_stem_none.fit_transform(train_stem_none.data)
    tfid_test_stem_none = tfid_stem_none.transform(test_stem_none.data)

    tfid_none_stop = TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')
    tfid_train_none_stop = tfid_none_stop.fit_transform(train_none_stop.data)
    tfid_test_none_stop = tfid_none_stop.transform(test_none_stop.data)

    count_none_none = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
    count_train_none_none = count_none_none.fit_transform(train_none_none.data)
    count_test_none_none = count_none_none.transform(test_none_none.data)

    count_stem_stop = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
    count_train_stem_stop = count_stem_stop.fit_transform(train_stem_stop.data)
    count_test_stem_stop = count_stem_stop.transform(test_stem_stop.data)

    count_stem_none = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
    count_train_stem_none = count_stem_none.fit_transform(train_stem_none.data)
    count_test_stem_none = count_stem_none.transform(test_stem_none.data)

    count_none_stop = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
    count_train_none_stop = count_none_stop.fit_transform(train_none_stop.data)
    count_test_none_stop = count_none_stop.transform(test_none_stop.data)

    # Train and Test
    res = {}
    algo_features = []

    algo_feature = 'NB,CountVectorizer'
    algo_features.append(algo_feature)
    clf = MultinomialNB()
    clf.fit(count_train_none_none, train_none_none.target)
    pred = clf.predict(count_test_none_none)
    res[algo_feature]={'pred': pred, 'target': test_none_none.target}

    algo_feature = 'NB,CountVectorizer+Stemming'
    algo_features.append(algo_feature)
    clf = MultinomialNB()
    clf.fit(count_train_stem_none, train_stem_none.target)
    pred = clf.predict(count_test_stem_none)
    res[algo_feature] = {'pred': pred, 'target': test_stem_none.target}

    algo_feature = 'NB,CountVectorizer+Lower case'
    algo_features.append(algo_feature)
    clf = MultinomialNB()
    clf.fit(count_train_none_stop, train_none_stop.target)
    pred = clf.predict(count_test_none_stop)
    res[algo_feature] = {'pred': pred, 'target': test_none_stop.target}

    algo_feature = 'NB,CountVectorizer+Stemming+Lower case'
    algo_features.append(algo_feature)
    clf = MultinomialNB()
    clf.fit(count_train_stem_stop, train_stem_stop.target)
    pred = clf.predict(count_test_stem_stop)
    res[algo_feature] = {'pred': pred, 'target': test_stem_stop.target}

    algo_feature = 'NB,TFIDFVectorizer'
    algo_features.append(algo_feature)
    clf = MultinomialNB()
    clf.fit(tfid_train_none_none, train_none_none.target)
    pred = clf.predict(tfid_test_none_none)
    res[algo_feature] = {'pred': pred, 'target': test_none_none.target}

    algo_feature = 'NB,TFIDFVectorizer+Stemming'
    algo_features.append(algo_feature)
    clf = MultinomialNB()
    clf.fit(tfid_train_stem_none, train_stem_none.target)
    pred = clf.predict(tfid_test_stem_none)
    res[algo_feature] = {'pred': pred, 'target': test_stem_none.target}

    algo_feature = 'NB,TFIDFVectorizer+Lower case'
    algo_features.append(algo_feature)
    clf = MultinomialNB()
    clf.fit(tfid_train_none_stop, train_none_stop.target)
    pred = clf.predict(tfid_test_none_stop)
    res[algo_feature] = {'pred': pred, 'target': test_none_stop.target}

    algo_feature = 'NB,TFIDFvectorizer+Stemming+Lower case'
    algo_features.append(algo_feature)
    clf = MultinomialNB()
    clf.fit(tfid_train_stem_stop, train_stem_stop.target)
    pred = clf.predict(tfid_test_stem_stop)
    res[algo_feature] = {'pred': pred, 'target': test_stem_stop.target}

    # Print result to output file
    for af in algo_features:
        val = res[af]
        precision = metrics.precision_score(val['target'], val['pred'], average='macro')
        recall = metrics.recall_score(val['target'], val['pred'], average='macro')
        f1_score = metrics.f1_score(val['target'], val['pred'], average='macro')
        out_file.write(str(af)+","+str(precision)+","+str(recall)+","+str(f1_score)+'\n')

    out_file.close()
