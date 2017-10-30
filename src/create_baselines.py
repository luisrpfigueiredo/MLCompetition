"""Sample script creating some baseline predictions."""
import os

import numpy as np

import data
import utils
import pandas
from sklearn import linear_model

NR_SUBJECTS = 8

UNIFORM_PREDICTIONS_BASENAME = os.path.join('Predictions', 'uniform')
AVG_PREDICTIONS_BASENAME = os.path.join('Predictions', 'average')


def predict_average(train_data, test_data):
    all_intervals = [
        interval for session in train_data for interval in session.intervals]
    targets = np.array([interval.session.subject for interval in all_intervals])

    targets_oh = utils.to_one_hot(targets, min_int=1, max_int=NR_SUBJECTS)
    avg = targets_oh.mean(axis=0)
    # Use this as the prediction for every test data sample
    predictions = [avg] * len(test_data)
    return predictions


def predict_uniform(test_data):
    predictions = np.ones(shape=(len(test_data), NR_SUBJECTS)) / NR_SUBJECTS
    return predictions


def main():
    dataset = data.load_pickled_data()
    train_data = dataset['train']
    test_data = dataset['test']

    predictions_unif = predict_uniform(test_data)
    pred_file_name = utils.generate_unqiue_file_name(
        UNIFORM_PREDICTIONS_BASENAME, 'npy')
    utils.dump_npy(predictions_unif, pred_file_name)
    print 'Dumped predictions to %s' % pred_file_name

    predictions_avg = predict_average(train_data, test_data)
    pred_file_name = utils.generate_unqiue_file_name(
        AVG_PREDICTIONS_BASENAME, 'npy')
    utils.dump_npy(predictions_avg, pred_file_name)
    print 'Dumped predictions to %s' % pred_file_name


if __name__ == '__main__':
    main()
