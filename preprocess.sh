#!/bin/bash

#!/bin/bash

# 配置文件的路径
config_file="/home/player/ControlNet/models/cldm_v15.yaml"

# 使用 sed 命令进行替换
sed -i 's/use_checkpoint: True/use_checkpoint: False/g' "$config_file"

# python3 torch2engine.py