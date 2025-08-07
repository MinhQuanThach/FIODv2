import os
import os.path as osp
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
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

def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

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
    yolo = YOLO('yolov8n.pt')
    yolo.to(args.gpu)
    yolo.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model = YOLO('yolov8n.pt').model
    model.train()
    model.cuda(args.gpu)

    # Initialize fog-pass filters (adjust input sizes based on YOLOv8n backbone)
    FogPassFilter1 = FogPassFilter_conv1(528)  # Placeholder; adjust per layer
    FogPassFilter2 = FogPassFilter_res1(2080)  # Placeholder; adjust per layer
    FogPassFilter1_optimizer = torch.optim.Adam(FogPassFilter1.parameters(), lr=5e-4)
    FogPassFilter2_optimizer = torch.optim.Adam(FogPassFilter2.parameters(), lr=1e-3)
    FogPassFilter1.cuda(args.gpu)
    FogPassFilter2.cuda(args.gpu)
    fogpassfilter_loss = FogPassFilterLoss(margin=0.1)

    # Data loaders
    cwsf_dataset = PairedCityscapes(args.data_dir, set=args.set, max_iters=args.num_steps * args.batch_size, img_size=args.img_size)
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
    opts = make_list(optimiser)

    # Feature extraction hooks
    feature_layers = [2, 4]  # Adjust based on YOLOv8n architecture
    features = {idx: [] for idx in feature_layers}
    handles = []

    def hook_fn(layer_idx):
        def hook(module, input, output):
            features[layer_idx].append(output.detach())
        return hook

    for layer_idx in feature_layers:
        handle = model.model[layer_idx].register_forward_hook(hook_fn(layer_idx))
        handles.append(handle)

    # Training loop
    for i_iter in tqdm(range(args.num_steps)):
        loss_det_cw_value = 0
        loss_det_sf_value = 0
        loss_fsm_value = 0
        loss_con_value = 0

        for opt in opts:
            opt.zero_grad()
        FogPassFilter1_optimizer.zero_grad()
        FogPassFilter2_optimizer.zero_grad()

        for sub_i in range(args.iter_size):
            # Step 1: Train fog-pass filters

            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            for param in FogPassFilter1.parameters():
                param.requires_grad = True
            for param in FogPassFilter2.parameters():
                param.requires_grad = True

            # Get batches
            batch_cwsf = next(cwsf_loader_iter, None)
            if batch_cwsf is None:
                cwsf_loader_iter = iter(cwsf_loader)
                batch_cwsf = next(cwsf_loader_iter)
            cw_img, sf_img, cw_label, sf_label, _= batch_cwsf
            cw_label = [{k: v.cuda(args.gpu) for k, v in label.items()} for label in cw_label]
            sf_label = [{k: v.cuda(args.gpu) for k, v in label.items()} for label in sf_label]

            batch_rf = next(rf_loader_iter, None)
            if batch_rf is None:
                rf_loader_iter = iter(rf_loader)
                batch_rf = next(rf_loader_iter)
            rf_img, _ = batch_rf

            sf_img, cw_img, rf_img = (Variable(sf_img).cuda(args.gpu),
                                      Variable(cw_img).cuda(args.gpu),
                                      Variable(rf_img).cuda(args.gpu))

            # Forward passes
            for key in features:
                features[key].clear()
            _ = model(cw_img)
            _ = model(sf_img)
            _ = model(rf_img)
            features_cw = {idx: features[idx][0] for idx in feature_layers}
            features_sf = {idx: features[idx][1] for idx in feature_layers}
            features_rf = {idx: features[idx][2] for idx in feature_layers}

            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            cw_features = {'layer0': features_cw[2], 'layer1': features_cw[4]}
            sf_features = {'layer0': features_sf[2], 'layer1': features_sf[4]}
            rf_features = {'layer0': features_rf[2], 'layer1': features_rf[4]}

            total_fpf_loss = 0

            for idx, layer in enumerate(fsm_weights):
                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.train()
                fogpassfilter_optimizer.zero_grad()

                sf_gram = [0] * args.batch_size
                cw_gram = [0] * args.batch_size
                rf_gram = [0] * args.batch_size
                fog_factor_sf = [0] * args.batch_size
                fog_factor_cw = [0] * args.batch_size
                fog_factor_rf = [0] * args.batch_size

                for batch_idx in range(args.batch_size):
                    sf_gram[batch_idx] = gram_matrix(sf_features[layer][batch_idx])
                    cw_gram[batch_idx] = gram_matrix(cw_features[layer][batch_idx])
                    rf_gram[batch_idx] = gram_matrix(rf_features[layer][batch_idx])

                    vector_sf_gram = sf_gram[batch_idx][torch.triu(torch.ones_like(sf_gram[batch_idx])) == 1]
                    vector_cw_gram = cw_gram[batch_idx][torch.triu(torch.ones_like(cw_gram[batch_idx])) == 1]
                    vector_rf_gram = rf_gram[batch_idx][torch.triu(torch.ones_like(rf_gram[batch_idx])) == 1]

                    fog_factor_sf[batch_idx] = fogpassfilter(vector_sf_gram)
                    fog_factor_cw[batch_idx] = fogpassfilter(vector_cw_gram)
                    fog_factor_rf[batch_idx] = fogpassfilter(vector_rf_gram)

                fog_factor_embeddings = torch.cat([torch.unsqueeze(f, 0) for tri in zip(fog_factor_sf, fog_factor_cw, fog_factor_rf) for f in tri], 0)
                fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
                fog_factor_embeddings = fog_factor_embeddings.div(fog_factor_embeddings_norm.unsqueeze(1))
                fog_factor_labels = torch.LongTensor([0, 1, 2] * args.batch_size).to(args.gpu)
                fog_pass_filter_loss = fogpassfilter_loss(fog_factor_embeddings, fog_factor_labels)
                total_fpf_loss += fog_pass_filter_loss

            total_fpf_loss.backward()

            # Step 2: Train YOLO model
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            for param in FogPassFilter1.parameters():
                param.requires_grad = False
            for param in FogPassFilter2.parameters():
                param.requires_grad = False

            loss_det_cw, loss_det_sf, loss_con, loss_fsm = 0, 0, 0, 0
            for key in features:
                features[key].clear()

            if i_iter % 3 == 0: # CW & SF
                det_cw = model(cw_img)
                det_sf = model(sf_img)
                if all(label['boxes'].numel() > 0 for label in cw_label):
                    batch_cw = convert_labels_to_ultralytics_format(cw_label)
                    _, loss_components = yolo.loss(batch_cw, det_cw)
                    loss_det_cw = loss_components.sum()
                else:
                    loss_det_cw = 0

                if all(label['boxes'].numel() > 0 for label in sf_label):
                    batch_sf = convert_labels_to_ultralytics_format(sf_label)
                    _, loss_components = yolo.loss(batch_sf, det_sf)
                    loss_det_sf = loss_components.sum()
                else:
                    loss_det_sf = 0

                # Consistency loss (simplified IoU)
                if det_cw[0].numel() > 0 and det_sf[0].numel() > 0:
                    loss_con = 1 - compute_iou(det_cw[0][:, :4], det_sf[0][:, :4]).mean()
                    # loss_con = torch.tensor(loss_con.mean().item(), device=det_cw[0].device, requires_grad=True)
                else:
                    loss_con = 0

                cw_features = {'layer0': features[2][0], 'layer1': features[4][0]}
                sf_features = {'layer0': features[2][1], 'layer1': features[4][1]}
                a_features, b_features = cw_features, sf_features

            if i_iter % 3 == 1: # SF & RF
                det_sf = model(sf_img)
                det_rf = model(rf_img)
                if all(label['boxes'].numel() > 0 for label in sf_label):
                    batch_sf = convert_labels_to_ultralytics_format(sf_label)
                    _, loss_components = yolo.loss(batch_sf, det_sf)
                    loss_det_sf = loss_components.sum()
                else:
                    loss_det_sf = 0

                sf_features = {'layer0': features[2][0], 'layer1': features[4][0]}
                rf_features = {'layer0': features[2][1], 'layer1': features[4][1]}
                a_features, b_features = rf_features, sf_features

            if i_iter % 3 == 2: # CW & RF
                det_cw = model(cw_img)
                det_rf = model(rf_img)
                if all(label['boxes'].numel() > 0 for label in cw_label):
                    batch_cw = convert_labels_to_ultralytics_format(cw_label)
                    _, loss_components = yolo.loss(batch_cw, det_cw)
                    loss_det_cw = loss_components.sum()
                else:
                    loss_det_cw = 0

                cw_features = {'layer0': features[2][0], 'layer1': features[4][0]}
                rf_features = {'layer0': features[2][1], 'layer1': features[4][1]}
                a_features, b_features = rf_features, cw_features

            for idx, layer in enumerate(fsm_weights):
                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.eval()
                layer_fsm_loss = 0

                for batch_idx in range(args.batch_size):
                    a_gram = gram_matrix(a_features[layer][batch_idx])
                    b_gram = gram_matrix(b_features[layer][batch_idx])
                    _, _, ha, wa = a_features[layer].size()
                    _, _, hb, wb = b_features[layer].size()

                    if i_iter % 3 in [1, 2]:
                        a_gram = a_gram * (hb * wb) / (ha * wa)

                    vector_a = a_gram[torch.triu(torch.ones_like(a_gram)) == 1]
                    vector_b = b_gram[torch.triu(torch.ones_like(b_gram)) == 1]

                    fog_factor_a = fogpassfilter(vector_a)
                    fog_factor_b = fogpassfilter(vector_b)
                    half = int(fog_factor_b.shape[0] / 2)

                    layer_fsm_loss += fsm_weights[layer] * torch.mean((fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa))**2) / half

                loss_fsm += layer_fsm_loss / args.batch_size

            loss = (loss_det_sf + loss_det_cw + args.lambda_fsm * loss_fsm + args.lambda_con * loss_con) / args.iter_size
            loss.backward()

            if loss_det_cw != 0:
                loss_det_cw_value += loss_det_cw.item() / args.iter_size
            if loss_det_sf != 0:
                loss_det_sf_value += loss_det_sf.item() / args.iter_size
            if loss_fsm != 0:
                loss_fsm_value += loss_fsm.item() / args.iter_size
            if loss_con != 0:
                loss_con_value += loss_con.item() / args.iter_size

            for opt in opts:
                opt.step()
            FogPassFilter1_optimizer.step()
            FogPassFilter2_optimizer.step()
            scheduler.step()

            wandb.log({
                "loss_det_cw": loss_det_cw_value,
                "loss_det_sf": loss_det_sf_value,
                "fsm_loss": args.lambda_fsm * loss_fsm_value,
                "consistency_loss": args.lambda_con * loss_con_value,
                "total_loss": loss
            }, step=i_iter)

    # Cleanup hooks
    for handle in handles:
        handle.remove()


if __name__ == '__main__':
    main()