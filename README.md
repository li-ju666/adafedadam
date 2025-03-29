# Accelerating Fair Federated Learning: Adaptive Federated Adam

This repository contains the implementation for the publication [Accelerating Fair Federated Learning: Adaptive Federated Adam.](https://ieeexplore.ieee.org/abstract/document/10584508/references#references) The goal of this work is to address fairness and efficiency challenges in federated learning with an adaptive optimization strategy.

The repository includes code to generate data, run training for the Adaptive Federated Adam method, and configure experiments via a centralized configuration file.

## Table of Contents
- [Installation](#installation)
- [Dataset Generation](#dataset-generation)
- [Training](#training)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/li-ju666/adafedadam.git
   cd adafedadam
   ```

2. Install the dependencies using `pip` and the provided `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Generation

To prepare and generate the dataset required for training, run:

```bash
python3 -m data_prepare.generate
```

Both the dataset generating script and training script will read configurations (e.g., data paths, parameters, etc.) from the `config.yaml` file.

## Training

Once the dataset is prepared, you can start training the model using the Adaptive Federated Adam method by running:

```bash
python3 main.py adafedadam
```

The training script will also load configurations from the `config.yml` file. To modify any experiment settings (hyperparameters, data parameters, model configurations, etc.), please update `config.yml` accordingly.

## Configuration

The main configuration for both dataset generation and training is located in the [`config.yml`](config.yml) file. You can adjust various settings such as:

- Dataset generating parameters.
- Federated learning settings (number of clients, local epochs, etc.).

## Citation

If you find this work useful for your research, please cite our paper using the following BibTeX entry:

```bibtex
@article{ju2024accelerating,
  title={Accelerating fair federated learning: Adaptive federated adam},
  author={Ju, Li and Zhang, Tianru and Toor, Salman and Hellander, Andreas},
  journal={IEEE Transactions on Machine Learning in Communications and Networking},
  year={2024},
  publisher={IEEE}
}
```

## License
This project is licensed under the MIT License
