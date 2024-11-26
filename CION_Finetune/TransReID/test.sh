#Single GPU
python test.py --config_file configs/market/vit_small.yml INPUT.SIZE_TRAIN [384,128] INPUT.SIZE_TEST [384,128] TEST.WEIGHT "path to test path"  MODEL.DEVICE_ID '("0")'


