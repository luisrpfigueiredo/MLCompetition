from sklearn import decomposition


def select_features(extracted_train_data):

    decomposer = decomposition.TruncatedSVD(n_components=10)
    decomposer.fit(extracted_train_data)

    print decomposer.explained_variance_
    print decomposer.explained_variance_ratio_
    print decomposer.singular_values_

    return decomposer
