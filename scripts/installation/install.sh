pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install tensorflow==2.15.0
pip install -U "jax[cuda12]"

cd ../..

cd third_party/dlimp
pip install -e .

cd ../lerobot
pip install -e .

cd ../openpi-client
pip install -e .

cd ../piper_sdk
pip install -e .

cd ../pika_sdk
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04/wxpython-4.2.3-cp311-cp311-linux_x86_64.whl
pip install -e .
cd ../..

cd src/training/openpi
pip install -e .
cd ../../..

cd scripts/installation
pip install -r requirements.txt
cd ../..