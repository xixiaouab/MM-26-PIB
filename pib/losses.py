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


def layer_scatter(features, labels):
    features = features.float()
    classes, inverse, counts = labels.unique(sorted=True, return_inverse=True, return_counts=True)
    layers, size = features.shape[0], classes.numel()
    means = features.new_zeros(layers, size, features.shape[-1]).index_add_(1, inverse, features)
    means = means / counts[None, :, None]
    deviations = (features - means[:, inverse]).square().sum(-1)
    within = features.new_zeros(layers, size).index_add_(1, inverse, deviations)
    valid = counts >= 2
    num_valid = int(valid.sum())
    zero = features.sum((1, 2)) * 0
    intra = within[:, valid].sum(1) / num_valid if num_valid else zero
    inter = (means[:, :, None] - means[:, None, :]).square().sum((1, 2, 3)) if size >= 2 else zero
    return intra, inter, num_valid, size


def scatter_statistics(features, labels):
    intra, inter, valid, classes = layer_scatter(features[None], labels)
    return intra[0], inter[0], valid, classes


def layer_sufficiency(features, labels, eps=1e-8):
    intra, inter, valid, classes = layer_scatter(features, labels)
    if not valid or classes < 2:
        return features.float().sum((1, 2)) * 0
    return intra / (inter + eps)


def sufficiency_loss(features, labels, eps=1e-8):
    return layer_sufficiency(features[None], labels, eps)[0]


def sufficiency_per_layer(pooled, labels, eps=1e-8):
    groups = {}
    for index, features in enumerate(pooled):
        groups.setdefault(features.shape[-1], []).append(index)
    values = [None] * len(pooled)
    for indices in groups.values():
        for index, value in zip(indices, layer_sufficiency(torch.stack([pooled[i] for i in indices]), labels, eps)):
            values[index] = value
    return torch.stack(values)


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
                 path_weight=0.1, routing=True, normalization="mean", eps=1e-8, label_smoothing=0.0):
        super().__init__()
        if depth < 1 or path_weight < 0 or eps <= 0 or not 0 <= label_smoothing < 1:
            raise ValueError("Invalid PIB depth, path weight, epsilon, or label smoothing.")
        self.label_smoothing = float(label_smoothing)
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
        self.active = bool(self.compression_weights.any() or self.sufficiency_weights.any() or self.path_weight > 0)

    def forward(self, output, labels, regularization_scale=1.0):
        if not self.active:
            logits = output["logits"] if isinstance(output, dict) else output
            task = F.cross_entropy(logits.float(), labels, label_smoothing=self.label_smoothing)
            return {"loss": task, "task": task}
        states = output["states"]
        if len(states) != self.alpha.numel():
            raise ValueError("Layer states and PIB schedules have different lengths.")
        raw_comp = torch.stack([compression_loss(state.tokens, self.eps) for state in states])
        raw_suff = sufficiency_per_layer([state.pooled for state in states], labels, self.eps)
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
        task = F.cross_entropy(output["logits"].float(), labels, label_smoothing=self.label_smoothing)
        regularizer = weighted.sum()
        total = task + regularization_scale * (regularizer + self.path_weight * path)
        return {"loss": total, "task": task, "pib": regularizer, "path": path,
                "compression": raw_comp.mean(), "sufficiency": raw_suff.mean(),
                "quality": quality, "layer_compression": raw_comp, "layer_sufficiency": raw_suff}
