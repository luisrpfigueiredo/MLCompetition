"""Script creating, saving and loading the data pickle file."""

import data


def main():
    print "about to do"
    data.create_pickled_data(overwrite_old=True)
    print "created data, loading dataset"
    dataset = data.load_pickled_data()
    print "dataset loaded"
    train_set = dataset['train']
    test_set = dataset['test']


if __name__ == '__main__':
    main()
