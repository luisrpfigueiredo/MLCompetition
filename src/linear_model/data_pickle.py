from individual.src import utils, data
import data_parser as parser


def main():
    print "Creating teacher's data if didn't exist"
    data.create_pickled_data(overwrite_old=False)

    print "Loading teacher's data"
    data_set = data.load_pickled_data()
    train_set = data_set['train']
    test_set = data_set['test']

    train_data, test_data = parser.parse_data(train_set, test_set)

    print len(train_data)
    print len(test_data)

    print "Saving cleaned, parsed version of pickled data"
    utils.dump_pickle(
        dict(train=train_data, test=test_data), data.PARSED_PICKLE_PATH)

    print "Saved to %s" % data.PARSED_PICKLE_PATH


if __name__ == '__main__':
    main()