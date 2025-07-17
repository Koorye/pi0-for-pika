#!/bin/bash
# 自动生成的设备绑定配置文件
# 生成时间: 2025-07-16 18:10:00

# 清空现有规则文件
sudo sh -c 'echo "" > /etc/udev/rules.d/serial.rules'
sudo sh -c 'echo "" > /etc/udev/rules.d/fisheye.rules'

# 串口设备绑定规则
sudo sh -c 'echo "ACTION==\"add\", KERNELS==\"1-4.4:1.0\", SUBSYSTEMS==\"usb\", MODE:=\"0777\", SYMLINK+=\"ttyUSB81\"" >> /etc/udev/rules.d/serial.rules'
sudo sh -c 'echo "ACTION==\"add\", KERNELS==\"1-4.4:1.0\", SUBSYSTEMS==\"usb\", MODE:=\"0777\", SYMLINK+=\"ttyUSB82\"" >> /etc/udev/rules.d/serial.rules'

# 鱼眼相机绑定规则
sudo sh -c 'echo "ACTION==\"add\", KERNEL==\"video[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]*\", KERNELS==\"1-4.3:1.0\", SUBSYSTEMS==\"usb\", MODE:=\"0777\", SYMLINK+=\"video81\"" >> /etc/udev/rules.d/fisheye.rules'
sudo sh -c 'echo "ACTION==\"add\", KERNEL==\"video[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]*\", KERNELS==\"1-2.3:1.0\", SUBSYSTEMS==\"usb\", MODE:=\"0777\", SYMLINK+=\"video82\"" >> /etc/udev/rules.d/fisheye.rules'

# 重新加载规则并触发
sudo udevadm control --reload-rules && sudo service udev restart && sudo udevadm trigger

echo "设备绑定规则已应用，请重新插拔设备以生效"
