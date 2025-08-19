from types import SimpleNamespace
from ultralytics import YOLO

from utils.train_config import get_arguments

args = get_arguments()

yolo = YOLO('yolov8n.pt')
yolo.to(args.gpu)
yolo.model.args = SimpleNamespace(box=0.05, cls=0.5, dfl=1.5)
model = YOLO('yolov8n.pt').model
model.train()
model.to(args.gpu)