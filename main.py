import yaml
import ujson
import argparse
from component.client import Client
from utils import preprocess
import torch
from component.trainer import FedAvg, FedAdam, AdaFedAdam


def main(config):

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    data_path = f"data/{config['dataset']}/partitioned.json"
    # prepare the data
    with open(data_path, 'r') as inf:
        data = ujson.load(inf)
    data = data['user_data']
    data = {idx: data[idx] for idx in list(data.keys())[:config['num_clients']]}

    # build dataset
    preprocessor = getattr(preprocess, config['dataset'])
    clients = [
        Client(
            k, config['dataset'], preprocessor(v),
            config['batch_size'], device) for k, v in data.items()]

    # build the trainer
    trainer_dict = {
        "fedavg": FedAvg,
        "fedadam": FedAdam,
        "adafedadam": AdaFedAdam
    }
    trainer = trainer_dict[config['trainer']](config['dataset'])

    # global training loop
    for round in range(config['num_rounds']):

        # evaluate the model
        if round % config['eval_every'] == 0:
            eval_result = trainer.eval(clients)
            # print(f"Evaluation result: {eval_result}")
            print(
                f"Round {round+1}/{config['num_rounds']} - "
                f"Avg acc: {eval_result['avg_acc']:.4f}, ")
                # f"Avg error: {eval_result['avg_error']:.4f}, "
                # f"Std acc: {eval_result['std_acc']:.4f}, "
                # f"Std error: {eval_result['std_error']:.4f}")

        # train the model
        trainer.train(
            clients, config['participation'], config['local_epochs'])


if __name__ == "__main__":
    # get the arguments from the config file
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trainer", type=str, default="fedavg",
        choices=["fedavg", "fedadam", "adafedadam"],
        help="The trainer to use")

    config['trainer'] = parser.parse_args().trainer

    main(config)
