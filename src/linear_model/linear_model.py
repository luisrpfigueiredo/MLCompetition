import os
import numpy as np
import individual.src.utils as utils
from sklearn import linear_model
from sklearn import svm

import individual.src.data as data

NR_SUBJECTS = 8
LINEAR_PREDICTIONS_BASENAME = os.path.join('Predictions', 'linear')


def predict_linear_model(train_data, test_data):
    """Do the magic"""
    print("training")

    model = train_model(train_data)

    dataset, users = build_sets(train_data)

    score = model.score(dataset, users)
    print("Accuracy on full set: {:.2f} %".format(score * 100))

    #k=20
    #print "Running k = {} cross valdiation".format(k)
    #k_fold_cv(model, dataset, users, k)


    print("predicting")
    return predict(model, test_data)


def train_model(train_data):
    """Train the model with the specified parsed train data"""

    dataset, users = build_sets(train_data)

    lin_model = linear_model.RidgeClassifierCV(fit_intercept=False)
    lin_model.fit(dataset, users)

    return lin_model


def predict(model, test_data):
    """ Create the predictions for the linear model with this test data"""
    print "starting predictions"
    predicts = []
    for interval in test_data:
        prediction = []
        original_predictions = list(model.predict(interval))
        for subject in range(1, NR_SUBJECTS + 1):
            prediction.append(original_predictions.count(subject) / (len(original_predictions) * 1.0))

        predicts.append(prediction)

    return predicts


def k_fold_cv(model, train_data, train_labels, k):
    """Run kfold cross validation and print the scores"""

    scores = [0] * k
    split_indexes = [0] * (k - 1)

    for temp in range(1, k):
        split_indexes[temp - 1] = int(temp * (len(train_data) / (1.0 * k)))

    print(split_indexes)

    train_folds = np.split(train_data, split_indexes)
    train_label_folds = np.split(train_labels, split_indexes)
    for iteration in range(0, len(train_folds)):
        test_subset = train_folds[iteration]
        test_labels = train_label_folds[iteration]

        temp_set = np.delete(train_folds, iteration, axis=0)
        temp_labels_set = np.delete(train_label_folds, iteration, axis=0)

        train_subset = temp_set[0]
        train_labels_set = temp_labels_set[0]

        for i in range(1, len(temp_set)):
            train_subset = np.append(train_subset, temp_set[i], axis=0)
            train_labels_set = np.append(train_labels_set, temp_labels_set[i], axis=0)

        model.fit(train_subset, train_labels_set)
        score = model.score(test_subset, test_labels)
        scores[iteration] = score

        print("Score for iteration = {} was: {:.2f} %".format(iteration, score * 100))

    print("Average score: {:.2f} %".format((sum(scores) * 100) / (1.0 * len(scores))))


def build_sets(training_data):
    x = []
    y = []

    for session in training_data:
        for sample in session.samples:
            x.append(sample)
            y.append(int(session.subject))

    return x, y


def main():
    dataset = data.load_pickled_data(data.PARSED_PICKLE_PATH)
    train_data = dataset['train']
    test_data = dataset['test']

    predictions_linear = predict_linear_model(train_data, test_data)
    print "Done"
    pred_file_name = utils.generate_unqiue_file_name(
        LINEAR_PREDICTIONS_BASENAME, 'npy')
    utils.dump_npy(predictions_linear, pred_file_name)
    print 'Dumped predictions to %s' % pred_file_name


if __name__ == '__main__':
    main()
