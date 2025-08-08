import torch
import torch.nn as nn

class ConsistencyLoss(nn.Module):
    def __init__(self, weight_box=1.0, weight_cls=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.weight_box = weight_box
        self.weight_cls = weight_cls

    def forward(self, pred_cw, pred_sf):
        """
        Args:
            pred_cw: (B, N, 4+C+1) YOLO raw outputs on clear-weather image
            pred_sf: (B, N, 4+C+1) YOLO raw outputs on synthetic-fog image

        Returns:
            scalar consistency loss
        """
        # Split into box and class components
        box_cw = pred_cw[..., :4]
        cls_cw = pred_cw[..., 5:]  # Skip objectness at index 4

        box_sf = pred_sf[..., :4]
        cls_sf = pred_sf[..., 5:]

        loss_box = self.mse(box_cw, box_sf)
        loss_cls = self.mse(cls_cw, cls_sf)

        return self.weight_box * loss_box + self.weight_cls * loss_cls
