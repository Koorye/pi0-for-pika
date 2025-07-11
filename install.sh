pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -U "jax[cuda12]"

cd third_party/dlimp
pip install -e .

cd ../lerobot
pip install -e .

cd ../openpi-client
pip install -e .

cd ../piper_sdk
pip install -e .

cd ../..

cd src/training/openpi
pip install -e .
cd ../../..

pip install -r requirements.txt