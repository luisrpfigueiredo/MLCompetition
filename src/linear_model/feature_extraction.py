from individual.src import utils, data
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


def extract_scaled_features(train_data, test_data):

    selected_train_features, selected_test_features = extract_features(train_data, test_data)
    scaled_train_features, scaled_test_features = scale_features(selected_train_features, selected_test_features)

    return scaled_train_features, scaled_test_features


def extract_features(train_data, test_data, number_components=5):
    all_samples = []
    for session in train_data:
        all_samples.extend(session.samples)

    decomposer = decomposition.FastICA(n_components=number_components)

    print "Decomposing train data"
    decomposer.fit(all_samples)

    train_data = [session._replace(samples=decomposer.transform(session.samples)) for session in train_data]

    print "Decomposing test data"
    test_data = [decomposer.transform(interval) for interval in test_data]

    return train_data, test_data


def scale_features(train_data, test_data):
    """Scales the features to a normalized curve"""

    all_samples = []
    for session in train_data:
        all_samples.extend(session.samples)

    scaler = StandardScaler()
    scaler.fit(all_samples)

    print "Scaling train data"
    train_data = [session._replace(samples=scaler.transform(session.samples)) for session in train_data]

    print "Scaling test data"
    test_data = [scaler.transform(interval) for interval in test_data]

    return train_data, test_data


def main():
    print "Loading parsed data"
    data_set = data.load_pickled_data(pickled_data_file_path=data.PARSED_PICKLE_PATH)
    train_set = data_set['train']
    test_set = data_set['test']

    train_data, test_data = extract_scaled_features(train_set, test_set)

    print "Saving cleaned, parsed version of pickled data"
    utils.dump_pickle(
        dict(train=train_data, test=test_data), data.PROCESSED_PICKLE_PATH)

    print "Saved to %s" % data.PROCESSED_PICKLE_PATH


if __name__ == '__main__':
    main()
