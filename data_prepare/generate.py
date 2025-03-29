import os
from data_prepare.partition import partition
from data_prepare.dataset import mnist, cifar10
import yaml
import ujson


data_funcs = {
    "mnist": mnist,
    "cifar10": cifar10,
}


def main():
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    path = f"data/{config['dataset']}"
    num_clients = config['num_clients']
    seed = config['seeds'][0]
    alpha = config['non_iid_level']

    if config['dataset'] in ["cifar10", "mnist"]:
        get_data = data_funcs[config['dataset']]
        trainx, trainy = get_data()

        # protocol:
        # partitioned_data is a dictionary
        # {
        #    'users': [list of user ids],
        #    'num_samples': [list of num_samples for each user],
        #    'partition_matrix': <the partition matrix>,
        #    'user_data': {
        #       'user_id': {'x': <the x data>, 'y': <the y data>},
        #     }
        # }
        data = partition(trainx, trainy, num_clients, alpha, seed)

        # save data to json file
        # create directory if not exists
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, 'partitioned.json')
        with open(file_path, 'w') as outfile:
            ujson.dump(data, outfile)

    else:
        # rename the json in the directory as 'partitioned.json'abs
        # list all json files in the directory
        json_files = [
            pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
        # rename the first json file as 'partitioned.json'
        if len(json_files) > 0:
            os.rename(os.path.join(path, json_files[0]),
                      os.path.join(path, 'partitioned.json'))
        else:
            raise FileNotFoundError("Custom dataset not found")
    return 0


if __name__ == '__main__':
    main()
