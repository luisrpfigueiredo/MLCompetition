import sklearn.preprocessing as preprocessing


def pre_process_data(train_set, test_set):
    print "Pre processing..."
    data_imputer = preprocessing.Imputer()

    print "Imputing train data"
    train_set = [session._replace(samples=data_imputer.fit_transform(session.samples)) for session in train_set]

    print "Imputing test data"
    test_set = [data_imputer.fit_transform(interval) for interval in test_set]

    return train_set, test_set
