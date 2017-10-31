import numpy as np
import collections

from data_preprocessing import pre_process_data
Session = collections.namedtuple(
    'Session',
    [
        'subject',  # The id of the subject
        'activity',  # The id of the subject
        'samples',  # All the samples
    ])


def parse_sample(raw_sample):
    """
   Parse a sample. This means cleaning infinity and nans to only nans.
   These nans will be treated during pre-processing in whatever means necessary
   :param raw_sample: sample to be parsed

   :return: the parsed sample
   """

    return [np.nan if np.isnan(value) or np.isinf(value) else value for value in raw_sample]


def parse_interval_data(raw_interval_data):
    """
   Parse one interval of data
   :param raw_interval_data: raw interval of data to be parsed.

   :return: the parsed interval data
   """

    return [parse_sample(raw_sample) for raw_sample in raw_interval_data]


def parse_train_data(raw_data, remove_overlap=True):
    """
   Parse a raw train data set.
   :param raw_data: data to be parsed
   :param remove_overlap: true if should remove overlap

   :return: array with Session Objects
   """

    print "Parsing train data"
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

    print "Parsing test data"
    parsed_data = [0] * len(raw_data)
    for idx, interval in enumerate(raw_data):
        parsed_data[idx] = parse_interval_data(interval.data)

    return parsed_data


def parse_data(raw_train_data, raw_test_data):
    """Does the parsing and pre-processing of the raw data"""
    print "Parsing data..."
    train_data = parse_train_data(raw_train_data, True)
    test_data = parse_test_data(raw_test_data)

    train_data, test_data = pre_process_data(train_data, test_data)

    return train_data, test_data
