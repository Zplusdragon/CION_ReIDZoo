import os
import time

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/market1501_mgn_R50_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/market1501_mgn_R50ibn_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/market1501_mgn_R101_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/market1501_mgn_R101ibn_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/market1501_mgn_R152_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/market1501_mgn_R152ibn_cion.yml"
print(command)
os.system(command)

#################################

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/msmt17_mgn_R50_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/msmt17_mgn_R50ibn_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/msmt17_mgn_R101_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/msmt17_mgn_R101ibn_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/msmt17_mgn_R152_cion.yml"
print(command)
os.system(command)

command = "CUDA_VISIBLE_DEVICE=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/msmt17_mgn_R152ibn_cion.yml"
print(command)
os.system(command)








