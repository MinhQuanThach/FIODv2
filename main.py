import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
from ultralytics import YOLO
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1, FogPassFilterLoss
from dataset.paired_cityscapes import PairedCityscapes
from dataset.foggy_zurich import FoggyZurich
from utils.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers
import wandb


def gram_matrix(tensor):
    """Compute Gram matrix for feature style comparison."""
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


def compute_iou(boxes1, boxes2):
    """Simple IoU computation for consistency loss."""
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)


def main():
    args = get_arguments()
    # Set random seeds for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # Initialize logging
    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'
    wandb.init(project='FIFO', name=run_name)
    wandb.config.update(args)

    # Enable CuDNN
    cudnn.enabled = True

    # Load YOLOv8n model
    model = YOLO('yolov8n.pt').model
    model.train()
    model.cuda(args.gpu)

    # Initialize fog-pass filters (adjust input sizes based on YOLOv8n backbone)
    FogPassFilter1 = FogPassFilter_conv1(2080)  # Placeholder; adjust per layer
    FogPassFilter2 = FogPassFilter_res1(8256)  # Placeholder; adjust per layer
    FogPassFilter1_optimizer = torch.optim.Adam(FogPassFilter1.parameters(), lr=5e-4)
    FogPassFilter2_optimizer = torch.optim.Adam(FogPassFilter2.parameters(), lr=1e-3)
    FogPassFilter1.cuda(args.gpu)
    FogPassFilter2.cuda(args.gpu)
    fogpassfilter_loss = FogPassFilterLoss(margin=0.1)

    # Data loaders
    cwsf_dataset = PairedCityscapes(args.data_dir, set=args.set, max_iters=args.num_steps * args.batch_size,img_size=args.img_size)
    cwsf_loader = data.DataLoader(
        cwsf_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=cwsf_dataset.collate_fn
    )
    rf_dataset = FoggyZurich(args.data_dir, set=args.set, max_iters=args.num_steps * args.batch_size, img_size=args.img_size)
    rf_loader = data.DataLoader(
        rf_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=rf_dataset.collate_fn
    )
    cwsf_loader_iter = iter(cwsf_loader)
    rf_loader_iter = iter(rf_loader)

    # Optimizer and scheduler
    optimiser = get_optimisers(model)
    scheduler = get_lr_schedulers(optimiser, args.num_steps)

    # Training loop
    for i_iter in tqdm(range(args.num_steps)):
        optimiser.zero_grad()
        FogPassFilter1_optimizer.zero_grad()
        FogPassFilter2_optimizer.zero_grad()

        # Load batches
        cw_img, sf_img, cw_label, sf_label, _ = next(cwsf_loader_iter)
        rf_img, _ = next(rf_loader_iter)
        cw_img, sf_img, rf_img = cw_img.cuda(args.gpu), sf_img.cuda(args.gpu), rf_img.cuda(args.gpu)
        cw_label = [{k: v.cuda(args.gpu) for k, v in label.items()} for label in cw_label]
        sf_label = [{k: v.cuda(args.gpu) for k, v in label.items()} for label in sf_label]

        # Step 1: Train Fog-Pass Filters
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        FogPassFilter1.train()
        FogPassFilter2.train()

        with torch.no_grad():
            features_cw = model(cw_img)[1]  # Backbone features
            features_sf = model(sf_img)[1]
            features_rf = model(rf_img)[1]

        # Extract features from specific layers (adjust indices as needed)
        for idx, (fogpassfilter, optimizer) in enumerate(
                [(FogPassFilter1, FogPassFilter1_optimizer), (FogPassFilter2, FogPassFilter2_optimizer)]):
            feature_idx = 2 if idx == 0 else 4  # Example layers; adjust based on YOLOv8n
            cw_gram = gram_matrix(features_cw[feature_idx])
            sf_gram = gram_matrix(features_sf[feature_idx])
            rf_gram = gram_matrix(features_rf[feature_idx])
            # Upper triangle vectors
            triu = torch.triu(torch.ones(cw_gram.size(0), cw_gram.size(1), device=args.gpu)) == 1
            fog_factors = torch.stack([
                fogpassfilter(cw_gram[triu]),
                fogpassfilter(sf_gram[triu]),
                fogpassfilter(rf_gram[triu])
            ])
            domain_labels = torch.tensor([1, 0, 2], device=args.gpu)  # CW=1, SF=0, RF=2
            fpf_loss = fogpassfilter_loss(fog_factors, domain_labels)
            fpf_loss.backward()
            optimizer.step()

        # Step 2: Train YOLO Model
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        FogPassFilter1.eval()
        FogPassFilter2.eval()

        # Cycle through CW, SF, RF for training
        if i_iter % 3 == 0:  # CW and SF
            det_cw = model(cw_img)
            det_sf = model(sf_img)
            if all(label['boxes'].numel() > 0 for label in cw_label):
                loss_det_cw = sum(model.module.loss(det_cw, cw_label)[1].values())
            else:
                loss_det_cw = 0

            if all(label['boxes'].numel() > 0 for label in sf_label):
                loss_det_sf = sum(model.module.loss(det_sf, sf_label)[1].values())
            else:
                loss_det_sf = 0

            # Consistency loss (simplified IoU)
            if det_cw[0].numel() > 0 and det_sf[0].numel() > 0:
                loss_con = 1 - compute_iou(det_cw[0][:, :4], det_sf[0][:, :4])
            else:
                loss_con = 0
            features_cw = model(cw_img)[1]
            features_sf = model(sf_img)[1]
            feats_a, feats_b = features_cw, features_sf
        elif i_iter % 3 == 1:  # SF and RF
            det_sf = model(sf_img)
            det_rf = model(rf_img)
            loss_det_cw = 0
            if all(label['boxes'].numel() > 0 for label in sf_label):
                loss_det_sf = sum(model.module.loss(det_sf, sf_label)[1].values())
            else:
                loss_det_sf = 0
            loss_con = 0
            features_sf = model(sf_img)[1]
            features_rf = model(rf_img)[1]
            feats_a, feats_b = features_rf, features_sf
        else:  # CW and RF
            det_cw = model(cw_img)
            det_rf = model(rf_img)
            if all(label['boxes'].numel() > 0 for label in cw_label):
                loss_det_cw = sum(model.module.loss(det_cw, cw_label)[1].values())
            else:
                loss_det_cw = 0
            loss_det_sf = 0
            loss_con = 0
            features_cw = model(cw_img)[1]
            features_rf = model(rf_img)[1]
            feats_a, feats_b = features_rf, features_cw

        # Fog Style Matching Loss
        loss_fsm = 0
        for idx, fogpassfilter in enumerate([FogPassFilter1, FogPassFilter2]):
            feature_idx = 2 if idx == 0 else 4
            a_gram = gram_matrix(feats_a[feature_idx])
            b_gram = gram_matrix(feats_b[feature_idx])
            triu = torch.triu(torch.ones(a_gram.size(0), a_gram.size(1), device=args.gpu)) == 1
            fog_factor_a = fogpassfilter(a_gram[triu])
            fog_factor_b = fogpassfilter(b_gram[triu])
            loss_fsm += torch.mean((fog_factor_a - fog_factor_b) ** 2)

        # Total loss
        loss = loss_det_cw + loss_det_sf + args.lambda_fsm*loss_fsm + args.lambda_con*loss_con
        loss.backward()
        optimiser.step()
        scheduler.step()

        # Logging
        wandb.log({
            'loss_det_cw': loss_det_cw,
            'loss_det_sf': loss_det_sf,
            'loss_fsm': args.lambda_fsm * loss_fsm,
            'loss_con': args.lambda_con * loss_con,
            'total_loss': loss
        }, step=i_iter)

        # Save checkpoints
        if i_iter % 1000 == 0 and i_iter > 0:
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, f'{run_name}_yolo_{i_iter}.pth'))
            torch.save(FogPassFilter1.state_dict(), osp.join(args.snapshot_dir, f'{run_name}_fogpass1_{i_iter}.pth'))
            torch.save(FogPassFilter2.state_dict(), osp.join(args.snapshot_dir, f'{run_name}_fogpass2_{i_iter}.pth'))


if __name__ == '__main__':
    main()