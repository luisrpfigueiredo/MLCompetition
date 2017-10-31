import numpy as np
import individual.src.data as data
import individual.src.utils as utils
import feature_selection


def feature_extraction(train_data, test_data):
    """ Extracts new features out of existing features using ICA """

    extracted_train_data = train_data
    extracted_test_data = test_data
    """ 
    x_hand_accel = raw_sample[0]
    y_hand_accel= raw_sample[1]
    z_hand_accel = raw_sample[2]
    x_hand_gyro = raw_sample[3]
    y_hand_gyro = raw_sample[4]
    z_hand_gyro = raw_sample[5]
    x_chest_accel = raw_sample[6]
    y_chest_accel = raw_sample[7]
    z_chest_accel = raw_sample[8]
    x_chest_gyro = raw_sample[9]
    y_chest_gyro = raw_sample[10]
    z_chest_gyro = raw_sample[11]

    hand_accel_norm = np.linalg.norm([x_hand_accel, y_hand_accel, z_hand_accel])
    chest_accel_norm = np.linalg.norm([x_chest_accel, y_chest_accel, z_chest_accel])
    hand_gyro_norm = np.linalg.norm([x_hand_gyro, y_hand_gyro, z_hand_gyro])
    chest_gyro_norm = np.linalg.norm([x_chest_gyro, y_chest_gyro, z_chest_gyro])

    difference_acceleration = abs(hand_accel_norm - chest_accel_norm)
    difference_angular_velocity = abs(hand_gyro_norm - chest_gyro_norm)

    hand_energy = abs(x_hand_accel) + abs(y_hand_accel) + abs(z_hand_accel) + \
                  abs(x_hand_gyro) + abs(y_hand_gyro) + abs(z_hand_gyro)

    chest_energy = abs(x_chest_accel) + abs(y_chest_accel) + abs(z_chest_accel) + \
                   abs(x_chest_gyro) + abs(y_chest_gyro) + abs(z_chest_gyro)

    new_sample = [
        x_hand_accel,
        y_hand_accel,
        z_hand_accel,
        #x_hand_gyro,
        #y_hand_gyro,
        #z_hand_gyro,
        x_chest_accel,
        y_chest_accel,
        z_chest_accel,
        #x_chest_gyro,
        #y_chest_gyro,
        #z_chest_gyro,
        hand_accel_norm,
        hand_gyro_norm,
        hand_accel_norm,
        chest_accel_norm,
        difference_acceleration,
        difference_angular_velocity,
        hand_energy,
        chest_energy
    ]
    """

    return extracted_train_data, extracted_test_data


def main():
    print "Loading parsed data"
    data_set = data.load_pickled_data(pickled_data_file_path=data.PARSED_PICKLE_PATH)
    train_set = data_set['train']
    test_set = data_set['test']

    train_data, test_data = feature_extraction(train_set, test_set)

    train_data, test_data = feature_selection.select_features(train_set, test_set)

    print "Saving cleaned, parsed version of pickled data"
    utils.dump_pickle(
        dict(train=train_data, test=test_data), data.PROCESSED_PICKLE_PATH)

    print "Saved to %s" % data.PROCESSED_PICKLE_PATH


if __name__ == '__main__':
    main()
