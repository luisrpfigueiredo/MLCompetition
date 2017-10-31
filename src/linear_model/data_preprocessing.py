import sklearn.preprocessing as preprocessing


def pre_process_data(train_data, test_data):
    print "Pre processing..."
    data_imputer = preprocessing.Imputer()

    print "Imputing train data"
    train_data = [interval._replace(samples=data_imputer.fit_transform(interval.samples)) for interval in train_data]

    print "Imputing test data"
    test_data = [data_imputer.fit_transform(interval) for interval in test_data]

    train_data, test_data = handle_outliers(train_data, test_data)

    return train_data, test_data


def handle_outliers(train_data, test_data):

    return train_data, test_data

    print "Handling train outliers"
    for index, session in enumerate(train_data):
        return

    print "Handling test outliers"

    return train_data, test_data
