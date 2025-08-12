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
    yolo.eval()
    FogPassFilter1.eval()
    FogPassFilter2.eval()

    # Validation datasets
    cwsf_val_dataset = PairedCityscapes(args.data_dir, set='val', img_size=args.img_size)
    rf_val_dataset = FoggyZurich(args.data_dir, set='val', img_size=args.img_size)

    cwsf_val_loader = DataLoader(
        cwsf_val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=cwsf_val_dataset.collate_fn
    )
    rf_val_loader = DataLoader(
        rf_val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=rf_val_dataset.collate_fn
    )

    # Metrics storage
    metrics = {'CW': {}, 'SF': {}, 'RF': {}}

    # Evaluate CW and SF (with labels)
    for domain in ['CW', 'SF']:
        total_loss = 0
        yolo_results = []
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

                # Collect predictions for metrics
                for pred in preds:
                    yolo_results.append(pred.cpu())

        # Compute YOLO metrics (mAP@50, mAP@50:95, etc.)
        yolo_metrics = yolo.val(data=yolo_results, task='detect')
        metrics[domain] = {
            'loss': total_loss / len(cwsf_val_loader),
            'mAP50': yolo_metrics.box.map50,
            'mAP50_95': yolo_metrics.box.map,
            'precision': yolo_metrics.box.p,
            'recall': yolo_metrics.box.r
        }

    # Evaluate RF (qualitative, no labels)
    rf_results = []
    for batch in rf_val_loader:
        rf_img, _ = batch
        rf_img = rf_img.to(args.gpu)
        with torch.no_grad():
            preds = model(rf_img)
            rf_results.extend([pred.cpu() for pred in preds])

    # Log qualitative RF metrics (e.g., number of detections)
    metrics['RF'] = {
        'num_detections': sum(len(pred.boxes) for pred in rf_results)
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


def plot_losses(run_name):
    """Plot training losses from wandb logs."""
    api = wandb.Api()
    run = api.run(f"FIFO/{run_name}")
    history = run.history()

    losses = {
        'loss_det_cw': [],
        'loss_det_sf': [],
        'fsm_loss': [],
        'consistency_loss': [],
        'total_loss': []
    }

    for row in history:
        for key in losses:
            if key in row and row[key] is not None:
                losses[key].append(row[key])

    plt.figure(figsize=(12, 8))
    for key, values in losses.items():
        if values:
            plt.plot(values, label=key)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)

    # Save plot to Google Drive
    plot_path = osp.join('/content/drive/MyDrive/FIOD_dataset/snapshots', f'{run_name}_losses.png')
    plt.savefig(plot_path)
    plt.close()

    # Log plot to wandb
    wandb.log({'training_losses_plot': wandb.Image(plot_path)})

    return plot_path


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