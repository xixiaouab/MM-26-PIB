import torch
from torch import nn
import torch.nn.functional as F


def compression_loss(tokens, eps=1e-8):
    """Equations (3)-(6): mean per-image off-diagonal correlation energy."""
    x = tokens.float()
    x = x / (x.norm(dim=-1, keepdim=True) + eps)
    n, d = x.shape[-2:]
    if n < d:
        energy = (x @ x.transpose(-2, -1)).square().sum((-1, -2)) / n ** 2
        diagonal = x.square().mean(-2).square().sum(-1)
        return (energy - diagonal).clamp_min(0).mean()
    correlation = x.transpose(-2, -1) @ x / n
    off_diagonal = correlation - torch.diag_embed(correlation.diagonal(dim1=-2, dim2=-1))
    return off_diagonal.square().sum((-1, -2)).mean()


def scatter_statistics(features, labels):
    features = features.float()
    centers, within = [], []
    for label in labels.unique(sorted=True):
        members = features[labels == label]
        center = members.mean(0)
        centers.append(center)
        if members.shape[0] >= 2:
            within.append((members - center).square().sum())
    zero = features.sum() * 0
    intra = torch.stack(within).mean() if within else zero
    if len(centers) >= 2:
        means = torch.stack(centers)
        inter = (means[:, None] - means[None, :]).square().sum()
    else:
        inter = zero
    return intra, inter, len(within), len(centers)


def sufficiency_loss(features, labels, eps=1e-8):
    intra, inter, valid, classes = scatter_statistics(features, labels)
    if not valid or classes < 2:
        return features.float().sum() * 0
    return intra / (inter + eps)


def layer_schedule(value, depth):
    if isinstance(value, dict):
        return torch.linspace(float(value["start"]), float(value["end"]), depth)
    if isinstance(value, (list, tuple)):
        if len(value) != depth:
            raise ValueError("A layer coefficient list must match the number of prompted layers.")
        return torch.tensor(value, dtype=torch.float32)
    return torch.full((depth,), float(value))


class PIBObjective(nn.Module):
    def __init__(self, depth, compression=None, sufficiency=None, alpha=1.0, delta=0.05,
                 path_weight=0.1, routing=True, normalization="mean", eps=1e-8):
        super().__init__()
        if depth < 1 or path_weight < 0 or eps <= 0:
            raise ValueError("Invalid PIB depth, path weight, or epsilon.")
        if normalization not in {"mean", "none"}:
            raise ValueError("normalization must be mean or none.")
        compression = {"start": 0.01, "end": 0.1} if compression is None else compression
        sufficiency = {"start": 0.1, "end": 0.01} if sufficiency is None else sufficiency
        for name, value in (("compression_weights", compression), ("sufficiency_weights", sufficiency),
                            ("alpha", alpha), ("delta", delta)):
            schedule = layer_schedule(value, depth)
            if (schedule < 0).any() or not torch.isfinite(schedule).all():
                raise ValueError(f"{name} must contain finite nonnegative coefficients.")
            self.register_buffer(name, schedule)
        self.routing_logits = nn.Parameter(torch.zeros(depth)) if routing else None
        self.path_weight = float(path_weight)
        self.normalization, self.eps = normalization, eps

    def forward(self, output, labels, regularization_scale=1.0):
        states = output["states"]
        if len(states) != self.alpha.numel():
            raise ValueError("Layer states and PIB schedules have different lengths.")
        raw_comp = torch.stack([compression_loss(state.tokens, self.eps) for state in states])
        raw_suff = torch.stack([sufficiency_loss(state.pooled, labels, self.eps) for state in states])
        comp, suff = raw_comp, raw_suff
        if self.normalization == "mean":
            comp = comp / comp.detach().mean().clamp_min(self.eps)
            suff = suff / suff.detach().mean().clamp_min(self.eps)
        if self.routing_logits is None:
            weighted = self.compression_weights * comp + self.sufficiency_weights * suff
        else:
            gates = self.routing_logits.sigmoid()
            weighted = gates * self.compression_weights * comp + (1 - gates) * self.sufficiency_weights * suff
        quality = comp + self.alpha * suff
        path = F.relu(quality[1:] - quality[:-1] - self.delta[:-1]).sum()
        task = F.cross_entropy(output["logits"].float(), labels)
        regularizer = weighted.sum()
        total = task + regularization_scale * (regularizer + self.path_weight * path)
        return {"loss": total, "task": task, "pib": regularizer, "path": path,
                "compression": raw_comp.mean(), "sufficiency": raw_suff.mean(),
                "quality": quality, "layer_compression": raw_comp, "layer_sufficiency": raw_suff}
