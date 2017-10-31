import numpy as np
import collections

from data_preprocessing import pre_process_data
Interval = collections.namedtuple(
    'Interval',
    [
        'subject',  # The id of the subject
        'activity',  # The id of the activity
        'samples',  # All the interval samples
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

   :return: array with parsed intervals
   """

    print "Parsing train data"
    if remove_overlap:
        parsed_intervals = [
            Interval(session.subject, int(session.activity), np.array(parse_interval_data(interval.data[len(interval.data) / 2:])))
            for session in raw_data
            for interval in session.intervals]
    else:
        parsed_intervals = [
            Interval(session.subject, int(session.activity), np.array(parse_test_data(interval.data)))
            for session in raw_data
            for interval in session.intervals]

    return parsed_intervals


def parse_test_data(raw_data):
    """
   Parse a raw test data set.
   :param raw_data: data to be parsed
   :return: array with parsed intervals
   """

    print "Parsing test data"
    parsed_data = [parse_interval_data(interval.data) for interval in raw_data]
    return parsed_data


def parse_data(raw_train_data, raw_test_data):
    """Does the parsing and pre-processing of the raw data"""
    print "Parsing data..."
    train_data = parse_train_data(raw_train_data, True)
    test_data = parse_test_data(raw_test_data)

    train_data, test_data = pre_process_data(train_data, test_data)

    return train_data, test_data
