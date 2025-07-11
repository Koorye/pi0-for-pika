# Pi-0 For Pika

A pipeline for Data Processing, Model Training, and Real-world Deployment for Pika.

## Installation

```bash
conda create -n pi0-for-pika python=3.11
conda activate pi0-for-pika
bash install.sh
```

## Usage

### Data Processing

1. Edit the configuration file `config.yaml` to set your parameters, such as `repo_id`, `data_root`, and `fps`.

2. Run the data processing script to prepare the dataset:
```bash
python scripts/data/pika2lerobot.py
```

3. After running the script, the processed lerobot dataset will be saved in the specified `data_root` directory.

### Model Training

1. Compute the dataset statistics:
```bash
python scripts/training/compute_norm_stats.py --config-name your_config
```

2. Train the model using the training script:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/training/train.py your_config --exp-name=your_experiment --overwrite
```

### Real-world Deployment

1. Deploy model server:
```bash
python scripts/deploy/run_server.py \
    --config your_training_config \
    --checkpoint your_model_checkpoint \
    --port 8000
```

2. Run the client to interact with the model server:
```bash
python scripts/deploy/run_client.py \
    --host 127.0.0.1 \
    --port 8000 \
    --can your_can_name
```