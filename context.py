import torch
from types import SimpleNamespace
from ultralytics import YOLO

from utils.train_config import get_arguments

args = get_arguments()
nc_new = 8
names = {0: 'person', 1: 'rider', 2: 'car', 3: 'bicycle', 4: 'motorcycle', 5: 'bus', 6: 'truck', 7: 'train'}

yolo = YOLO('/content/FIOD/yolov8n.yaml')
yolo.model.nc = nc_new
yolo.model.names = names

# load pretrained weights (from official .pt) into the new model, but allow missing keys
pretrained_path = 'content/FIOD/yolov8n.pt'
ckpt = torch.load(pretrained_path, map_location='cpu')
state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt

# load with strict=False so weights that don't match (detect head) are skipped
missing, unexpected = yolo.model.load_state_dict(state_dict, strict=False)
print("Loaded pretrained weights with strict=False; missing keys (skipped):", len(missing))
print("Unexpected keys (not used):", len(unexpected))

yolo.to(args.gpu)
yolo.model.args = SimpleNamespace(box=0.05, cls=0.5, dfl=1.5)

# model = YOLO('yolov8n.pt').model
# model.train()
# model.to(args.gpu)