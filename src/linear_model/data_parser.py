import numpy as np
import collections

from sklearn.preprocessing import StandardScaler

from feature_extraction import generate_sample
from feature_selection import select_features
Session = collections.namedtuple(
    'Session',
    [
        'subject',  # The id of the subject
        'activity',  # The id of the subject
        'samples',  # All the samples
    ])


def valid_coordinates(sample, fix_val=False):
    """
   Check if a given sample contains nulls, nans or infinite.
   :param sample: coordinates to check
   :param fix_val: true to replace the invalid values, false otherwise

   :return: true if valid, false otherwise. Still returns false if fixed
   """
    valid = True
    for index, value in enumerate(sample):
        if np.isinf(value) or np.isnan(value):
            valid = False
            if fix_val:
                sample[index] = 0

    return valid


def parse_sample(raw_sample, return_on_invalid=True, fix_invalid=False):
    """
   Parse a sample. Check if the sample is valid and then call feature selection
   on it.
   :param raw_sample: sample to be parsed
   :param return_on_invalid: true if should return None for invalid samples
   :param fix_invalid: true to fix the invalid points
   :return: the parsed sample if it was valid or None if not
   """

    if not valid_coordinates(raw_sample, fix_invalid) and return_on_invalid:
        return None

    # Run feature selection
    parsed_sample = generate_sample(raw_sample)
    #parsed_sample = raw_sample

    return parsed_sample


def parse_interval_data(raw_interval_data, return_on_invalid=True, fix_invalid=False):
    """
   Parse one interval of data. It will parse every sample of the interval
   :param raw_interval_data: raw interval of data to be parsed.
   :param return_on_invalid: true if should return None for invalid samples
   :param fix_invalid: true to fix the invalid points
   :return: the parsed interval data
   """
    parsed_interval_data = []

    for idx, raw_sample in enumerate(raw_interval_data):
        sample = parse_sample(raw_sample, return_on_invalid, fix_invalid)
        if sample is None:
            continue

        parsed_interval_data.append(sample)

    return parsed_interval_data


def parse_train_data(raw_data, remove_overlap=True):
    """
   Parse a raw train data set.
   :param raw_data: data to be parsed
   :param remove_overlap: true if should remove overlap

   :return: array with Session Objects
   """

    nr_sessions = len(raw_data)
    parsed_sessions_data = [0] * nr_sessions

    for session_index, session in enumerate(raw_data):
        flattened_intervals = []

        for interval_index, interval in enumerate(session.intervals):
            # Remove Overlap
            interval_data = interval.data
            if remove_overlap and not interval_index == 0:
                interval_data = interval.data[len(interval.data) / 2:]

            flattened_intervals.extend(parse_interval_data(interval_data))

        parsed_sessions_data[session_index] = \
            Session(session.subject, int(session.activity), np.array(flattened_intervals))

    return parsed_sessions_data


def parse_test_data(raw_data):
    """
   Parse a raw test data set.
   :param raw_data: data to be parsed
   :return: array with test sets
   """
    parsed_data = [0] * len(raw_data)
    for idx, interval in enumerate(raw_data):
        parsed_data[idx] = parse_interval_data(interval.data, True, True)

    return parsed_data


def parse_data(raw_train_data, raw_test_data):

    print "Parsing and executing feature extraction..."
    train_data = parse_train_data(raw_train_data, True)
    test_data = parse_test_data(raw_test_data)

    print "Aggregating samples"
    all_samples = []
    for session in train_data:
        all_samples.extend(session.samples)

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

    return train_data, test_data
