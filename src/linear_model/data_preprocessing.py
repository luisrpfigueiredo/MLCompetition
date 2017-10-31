import sklearn.preprocessing as preprocessing


def pre_process_data(train_set, test_set):
    print "Pre processing..."
    data_imputer = preprocessing.Imputer()

    for index, session in train_set:
        new_samples = data_imputer.fit_transform(session.samples)
        train_set[index] = session._replace(samples= new_samples)

    for index, interval in test_set:
        test_set[index] = data_imputer.fit_transform(interval)

    return train_set, test_set

"""

 print "Aggregating samples"


    print "Fitting Feature Selector"
    data_decomposer = select_features(all_samples)
    extracted_samples = data_decomposer.transform(all_samples)

    print "Fitting Feature Scaler"
    data_scaler = StandardScaler()
    data_scaler.fit(extracted_samples)

    print "Extracting and Rescaling train data"
    for index, session in enumerate(train_data):
        train_data[index] = session._replace(samples=
                                             data_scaler.transform(
                                                 data_decomposer.transform(
                                                     session.samples)
                                             )
        )

    print "Extracting and Rescaling test data"
    for index, interval in enumerate(test_data):
        test_data[index] = data_scaler.transform(data_decomposer.transform(interval))
"""