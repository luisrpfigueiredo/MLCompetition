from sklearn import decomposition


def select_features(train_data, test_data):

    selected_train_features = train_data
    selected_test_features = test_data

    all_samples = []
    for session in train_data:
        all_samples.extend(session.samples)

    decomposer = decomposition.TruncatedSVD(n_components=10)
    decomposer.fit(train_data)

    return selected_train_features, selected_test_features

