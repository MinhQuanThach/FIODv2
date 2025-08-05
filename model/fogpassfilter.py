import torch
import torch.nn as nn

class FogPassFilter_conv1(nn.Module):
    def __init__(self, inputsize):
        super(FogPassFilter_conv1, self).__init__()

        self.hidden = nn.Linear(inputsize, inputsize//2)
        self.hidden2 = nn.Linear(inputsize//2, inputsize//4)
        self.output = nn.Linear(inputsize//4, 64)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)

        return x

class FogPassFilter_res1(nn.Module):
    def __init__(self, inputsize):
        super(FogPassFilter_res1, self).__init__()

        self.hidden = nn.Linear(inputsize, inputsize//8)
        self.output = nn.Linear(inputsize//8, 64)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.output(x)

        return x

class FogPassFilterLoss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Compute pairwise cosine distances
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        cosine_dist = 1 - torch.mm(norm_embeddings, norm_embeddings.t())

        # Create mask for pairs from same domain (1) and different domains (0)
        labels_mat = labels.expand(len(labels), len(labels))
        same_domain_mask = (labels_mat == labels_mat.t()).float()

        # Compute positive pair loss: (1 - I(a,b))[m - d(F^a, F^b)]_+^2
        pos_pair_loss = (1 - same_domain_mask) * torch.pow(
            torch.clamp(self.margin - cosine_dist, min=0), 2
        )

        # Compute negative pair loss: I(a,b)[d(F^a, F^b) - m]_+^2
        neg_pair_loss = same_domain_mask * torch.pow(
            torch.clamp(cosine_dist - self.margin, min=0), 2
        )

        # Sum both terms and remove diagonal elements
        mask = 1 - torch.eye(len(labels), device=labels.device)
        total_loss = (pos_pair_loss + neg_pair_loss) * mask

        # Return mean over all valid pairs
        return total_loss.sum() / (mask.sum() + 1e-8)