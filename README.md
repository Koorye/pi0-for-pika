# Pi-0 For Pika

A pipeline for Data Processing, Model Training, and Real-world Deployment for Pika.

## Installation

```bash
git clone https://github.com/Koorye/pi0-for-pika.git
cd pi0-for-pika/scripts/installation
conda create -n pi0-for-pika python=3.11
conda activate pi0-for-pika
bash install.sh
```

After that, test GPUs by running: 
```bash
python scripts/installation/check_gpu_available.py
```

If all GPUs are available, this script will return:
```
GPU is available for PyTorch, TensorFlow, and JAX.
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

The pika config has been added to `src/training/openpi/src/openpi/training/config.py`, available configs include:

- `pi0_pika_lora`: fine-tuning with LoRA
- `pi0_pika`: full fine-tuning

Training hyperparameters:
| Hyperparameter | Value |
|----------------|-------|
| Batch Size     | 32    |
| Training Steps | 30000 |

⚠️ State token is replaced by a zero vector in the model, which is different from the original Pi-0 paper. 
This is due to the fact that the state token is not available in the Pika dataset. 
See the modification in line 217 of `src/training/openpi/src/openpi/models/pi0.py`.

⚠️ You need one GPU with at least 24GB memory to run the training.

1. Compute the dataset statistics:
```bash
python scripts/training/compute_norm_stats.py --config-name your_config
```

2. Train the model using the training script:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/training/train.py your_config --exp-name=your_experiment --overwrite
```

(Optional) If you want to add your own training config, please follow the format in:

- `src/training/openpi/src/openpi/training/config.py`
- `src/training/openpi/src/openpi/policies`

and add your config to the `_CONFIGS` list.

### Real-world Deployment

1. Deploy model server:
```bash
python scripts/deploy/run_server.py \
    --config your_training_config \
    --checkpoint your_model_checkpoint \
    --port 8000
```

2. Connect usb to your robot arm and check all cans are available:
```bash
bash scripts/deploy/find_all_can_port.sh
```
The result will be similar to:
```bash
Both ethtool and can-utils are installed.
Interface can0 is connected to USB port 3-1.4:1.0
Interface can1 is connected to USB port 3-1.1:1.0 # for multi-arm setup, if you have two arms connected
```

3. Edit `scripts/deploy/can_muti_activate.sh` to update the USB_PORTS variable with the ports found in the previous step. For example:
```bash
USB_PORTS["3-1.4:1.0"]="can_left:1000000"
USB_PORTS["3-1.1:1.0"]="can_right:1000000" # for multi-arm setup, if you have two arms connected
```
Then run the script to activate the CAN interfaces:
```bash
bash scripts/deploy/can_activate.sh can_piper 1000000 "3-1.4:1.0" # for single-arm setup
bash scripts/deploy/can_muti_activate.sh # for multi-arm setup
```

4. Run `ifconfig` to check if all can interfaces are available.

5. Run the client to interact with the model server:

(single-arm)
```bash
python scripts/deploy/run_client.py \
    --host 127.0.0.1 \
    --port 8000 \
    --can your_can_name
```

(multi-arm)
```bash
python scripts/deploy/run_multi_arm_client.py \
    --host 127.0.0.1 \
    --port 8000 \
    --left-can your_can_name_left \
    --right-cam your_can_name_right
```