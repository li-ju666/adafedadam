import numpy as np
from math import floor
import random


# pretty print a matrix
def pretty_print_matrix(matrix):
    """
    Pretty print a matrix
    :param matrix: input matrix
    :return: None
    """
    for row in matrix:
        for col in row:
            print('{:.2f}'.format(col), end=' ')
        print()


def partition(xs, ys, num_clients,
              non_iid_level, seed):
    """
    Partition the data into clients
    if non_iid_level is 0, then the data is iid, otherwise
    dirichlet distribution is used to generate non-iid data.
    :param xs: input data
    :param ys: input labels
    :param num_clients: number of clients
    :param non_iid_level: non-iid level
    :return: partitioned data
    """
    # set random seed
    random.seed(seed)
    np.random.seed(seed)

    num_classes = np.unique(ys).shape[0]

    # generate partition matrix to partition data for clients
    if non_iid_level:
        # generate partition matrix with dirichlet distribution
        too_few_data = True
        while too_few_data:
            partition_matrix = np.random.dirichlet(
                (non_iid_level, )*num_clients, num_classes)

            # ensure that each partition has at least
            # \(\frac{1}{10 * num_clients}\) of all data
            too_few_data = any(np.sum(partition_matrix, axis=0) <
                               1 / (10 * num_clients))
    else:
        # generate partition matrix with uniform distribution
        partition_matrix = np.ones((num_classes, num_clients)) / num_clients

    # print partition matrix
    print("Partition matrix:")
    pretty_print_matrix(partition_matrix.transpose())

    # partition data
    user_data = {idx: {'x': [], 'y': []}
                 for idx in range(num_clients)}

    for label_idx, label in enumerate(np.unique(ys)):
        xs_of_y = xs[ys == label]

        num_samples = xs_of_y.shape[0]

        # shuffle the data
        np.random.shuffle(xs_of_y)

        for client_idx in range(num_clients):
            local_size = floor(
                partition_matrix[label_idx, client_idx] * num_samples)

            x = xs_of_y[:local_size]
            xs_of_y = xs_of_y[local_size:]
            y = [label]*local_size

            user_data[client_idx]['x'].append(x)
            user_data[client_idx]['y'].append(y)

    # concatenate all data
    for each in user_data:
        user_data[each]['x'] = np.concatenate(user_data[each]['x'])
        user_data[each]['y'] = np.concatenate(user_data[each]['y'])

    for each in user_data:
        # shuffle client datasets
        idx = np.random.permutation(user_data[each]['y'].shape[0])
        user_data[each]['x'] = user_data[each]['x'][idx]
        user_data[each]['y'] = user_data[each]['y'][idx]

    # convert all numpy arrays to lists
    for each in user_data:
        user_data[each]['x'] = user_data[each]['x'].tolist()
        user_data[each]['y'] = user_data[each]['y'].tolist()

    users = list(user_data.keys())
    # return partitioned data
    data = {
        "users": users,
        'num_samples': [len(user_data[each]['y']) for each in users],
        'partition_matrix': partition_matrix.tolist(),
        "user_data": user_data,
    }
    return data


# test code
if __name__ == '__main__':
    sample_size = 10000
    xs = np.ones((sample_size, 20))
    ys = np.random.randint(0, 10, sample_size)
    partitioned_data = partition(xs, ys, 5, 100, 0.1, 42)
