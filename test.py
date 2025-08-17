import os

import torch
from ultralytics import YOLO
import wandb
import matplotlib.pyplot as plt
import os.path as osp
from dataset.paired_cityscapes import PairedCityscapes
from dataset.foggy_zurich import FoggyZurich
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import yaml

def make_temp_yaml(dataset_dict, save_path="temp_data.yaml"):
    with open(save_path, 'w') as f:
        yaml.dump(dataset_dict, f)
    return save_path

def convert_labels_to_ultralytics_format(label_list):
    cls_list = []
    bbox_list = []
    batch_idx_list = []

    for i, label in enumerate(label_list):
        if label['boxes'].numel() == 0:
            continue
        cls_list.append(label['labels'])  # shape: [num_objs]
        bbox_list.append(label['boxes'])  # shape: [num_objs, 4]
        batch_idx_list.append(torch.full((label['labels'].shape[0],), i, device=label['labels'].device, dtype=torch.long))

    if not cls_list:
        return None  # means empty labels

    return {
        'cls': torch.cat(cls_list, dim=0),
        'bboxes': torch.cat(bbox_list, dim=0),
        'batch_idx': torch.cat(batch_idx_list, dim=0),
    }

def test_model(args, model, yolo, FogPassFilter1, FogPassFilter2):
    """Test FIOD model on validation sets (CW, SF, RF) using YOLO metrics."""
    model.eval()
    yolo.model.load_state_dict(model.state_dict()) # Update yolo weights with trained model
    yolo.eval()
    FogPassFilter1.eval()
    FogPassFilter2.eval()

    # Validation datasets
    cwsf_val_dataset = PairedCityscapes(args.data_dir, set='val', img_size=args.img_size)

    cwsf_val_loader = DataLoader(
        cwsf_val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=cwsf_val_dataset.collate_fn
    )

    # Metrics storage
    metrics = {'CW': {}, 'SF': {}, 'RF': {}}
    dataset_dict = {'CW': {}, 'SF': {}, 'RF': {}}

    # Dataset dictionaries for YOLO's .val()
    dataset_dict['CW'] = {
        'path': '/content/drive/MyDrive/FIOD_dataset/data/CW',
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'person', 1: 'rider', 2: 'car', 3: 'bicycle', 4: 'motorcycle', 5: 'bus', 6: 'truck', 7: 'train'}
    }
    dataset_dict['SF'] = {
        'path': '/content/drive/MyDrive/FIOD_dataset/data/SF',
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'person', 1: 'rider', 2: 'car', 3: 'bicycle', 4: 'motorcycle', 5: 'bus', 6: 'truck', 7: 'train'}
    }

    # Evaluate CW and SF (with labels)
    for domain in ['CW', 'SF']:
        total_loss = 0
        for batch in cwsf_val_loader:
            cw_img, sf_img, cw_label, sf_label, _, _, _ = batch
            img = cw_img if domain == 'CW' else sf_img
            label = cw_label if domain == 'CW' else sf_label
            img = img.to(args.gpu)
            label = [{k: v.to(args.gpu) for k, v in l.items()} for l in label]

            # Convert labels to Ultralytics format
            batch_label = convert_labels_to_ultralytics_format(label)
            if batch_label is None:
                continue

            with torch.no_grad():
                preds = model(img)
                loss_components, _ = yolo.loss(batch_label, preds)
                total_loss += loss_components.sum().item()

        yaml_path = make_temp_yaml(dataset_dict[domain], "data.yaml")
        yolo_metrics = yolo.val(data=yaml_path, task='detect')
        metrics[domain] = {
            'loss': total_loss / len(cwsf_val_loader),
            'mAP50': yolo_metrics.box.map50,
            'mAP50_95': yolo_metrics.box.map,
            'precision': yolo_metrics.box.p,
            'recall': yolo_metrics.box.r
        }

    # Log qualitative RF metrics (e.g., number of detections)
    pred_results = yolo.predict('/content/drive/MyDrive/FIOD_dataset/data/RF/images/val', save=False)
    metrics['RF'] = {
        'num_detections': sum(len(pred.boxes) for pred in pred_results)
    }

    # Log metrics to wandb
    wandb.log({
        'val/CW_loss': metrics['CW']['loss'],
        'val/CW_mAP50': metrics['CW']['mAP50'],
        'val/CW_mAP50_95': metrics['CW']['mAP50_95'],
        'val/SF_loss': metrics['SF']['loss'],
        'val/SF_mAP50': metrics['SF']['mAP50'],
        'val/SF_mAP50_95': metrics['SF']['mAP50_95'],
        'val/RF_num_detections': metrics['RF']['num_detections']
    })

    return metrics


def save_model(args, model, FogPassFilter1, FogPassFilter2, run_name, i_iter):
    """Save model and filter weights to Google Drive."""
    snapshot_dir = osp.join(args.snapshot_dir, run_name)
    os.makedirs(snapshot_dir, exist_ok=True)

    # Save YOLO model
    torch.save(model.state_dict(), osp.join(snapshot_dir, f'yolo_{i_iter}.pth'))
    # Save fog-pass filters
    torch.save(FogPassFilter1.state_dict(), osp.join(snapshot_dir, f'fogpass1_{i_iter}.pth'))
    torch.save(FogPassFilter2.state_dict(), osp.join(snapshot_dir, f'fogpass2_{i_iter}.pth'))

    return snapshot_dir


def load_model(args, model, FogPassFilter1, FogPassFilter2, run_name, i_iter):
    """Load model and filter weights from Google Drive."""
    snapshot_dir = osp.join(args.snapshot_dir, run_name)

    model.load_state_dict(torch.load(osp.join(snapshot_dir, f'yolo_{i_iter}.pth')))
    FogPassFilter1.load_state_dict(torch.load(osp.join(snapshot_dir, f'fogpass1_{i_iter}.pth')))
    FogPassFilter2.load_state_dict(torch.load(osp.join(snapshot_dir, f'fogpass2_{i_iter}.pth')))

    return model, FogPassFilter1, FogPassFilter2