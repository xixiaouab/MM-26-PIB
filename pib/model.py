from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LayerState:
    index: int
    tokens: torch.Tensor
    pooled: torch.Tensor


def _partition(x, size):
    b, h, w, c = x.shape
    wh, ww = size
    return x.reshape(b, h // wh, wh, w // ww, ww, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, wh * ww, c)


def _reverse(x, size, h, w):
    wh, ww = size
    return x.reshape(-1, h // wh, w // ww, wh, ww, x.shape[-1]).permute(0, 1, 3, 2, 4, 5).reshape(-1, h, w, x.shape[-1])


def prompted_swin_block(block, x, prompt):
    if prompt is None:
        return block(x)
    batch, height, width, dim = x.shape
    shifted = torch.roll(block.norm1(x), shifts=tuple(-i for i in block.shift_size), dims=(1, 2))
    wh, ww = block.window_size
    shifted = F.pad(shifted, (0, 0, 0, (-width) % ww, 0, (-height) % wh))
    hp, wp = shifted.shape[1:3]
    windows = _partition(shifted, block.window_size)
    count = windows.shape[0] // batch
    prompts = block.norm1(prompt)[:, None].expand(-1, count, -1, -1).reshape(-1, prompt.shape[1], dim)
    windows = torch.cat((prompts, windows), dim=1)
    attn = block.attn
    q, k, v = attn.qkv(windows).reshape(windows.shape[0], windows.shape[1], 3,
                                      attn.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
    scores = (q * attn.scale) @ k.transpose(-2, -1)
    p = prompt.shape[1]
    scores = scores + F.pad(attn._get_rel_pos_bias(), (p, 0, p, 0))
    mask = block.get_attn_mask(shifted) if getattr(block, "dynamic_mask", False) else block.attn_mask
    if mask is not None:
        mask = F.pad(mask, (p, 0, p, 0))
        n = windows.shape[1]
        scores = (scores.reshape(-1, mask.shape[0], attn.num_heads, n, n)
                  + mask[None, :, None]).reshape(-1, attn.num_heads, n, n)
    output = (attn.attn_drop(scores.softmax(-1)) @ v).transpose(1, 2).reshape(windows.shape)
    output = attn.proj_drop(attn.proj(output))[:, p:]
    output = _reverse(output, block.window_size, hp, wp)[:, :height, :width]
    output = torch.roll(output, shifts=block.shift_size, dims=(1, 2))
    x = x + block.drop_path1(output)
    flat = x.reshape(batch, height * width, dim)
    return (flat + block.drop_path2(block.mlp(block.norm2(flat)))).reshape(batch, height, width, dim)


class LowRank(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout=0.0):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(self.dropout(x))) * self.scale


class Adapter(nn.Module):
    def __init__(self, width, dim, scale):
        super().__init__()
        self.down = nn.Linear(width, dim)
        self.up = nn.Linear(dim, width)
        self.scale = scale
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.relu(self.down(x))) * self.scale


class ScaleShift(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.shift = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        return x * self.scale + self.shift


LORA_TARGETS = {"qkv": lambda block: block.attn.qkv, "proj": lambda block: block.attn.proj,
                "fc1": lambda block: block.mlp.fc1, "fc2": lambda block: block.mlp.fc2}
SSF_TARGETS = {"norm1": lambda block: block.norm1, "qkv": lambda block: block.attn.qkv,
               "proj": lambda block: block.attn.proj, "norm2": lambda block: block.norm2,
               "fc1": lambda block: block.mlp.fc1, "fc2": lambda block: block.mlp.fc2}


def _add_to_output(module):
    def hook(layer, inputs, output):
        return output + module(inputs[0])
    return hook


def _transform_output(module):
    def hook(layer, inputs, output):
        return module(output)
    return hook


class PromptedVisionTransformer(nn.Module):
    def __init__(self, backbone, num_classes, prompt_length=20, prompt_layers="all", prompt_std=0.02,
                 prompt_dropout=0.0, prompt_init="normal", features="cls", lora=None, adapter=None, ssf=False):
        super().__init__()
        if min(num_classes, prompt_length) < 1 or prompt_std <= 0:
            raise ValueError("Class count, prompt length, and initialization std must be positive.")
        if features not in {"cls", "cls_gap"} or prompt_init not in {"normal", "xavier"}:
            raise ValueError("features must be cls or cls_gap; prompt_init must be normal or xavier.")
        self.backbone = backbone.requires_grad_(False)
        self.is_swin = hasattr(backbone, "layers")
        if self.is_swin:
            blocks = [block for stage in backbone.layers for block in stage.blocks]
            dimensions = [block.attn.qkv.in_features for block in blocks]
        elif hasattr(backbone, "blocks") and getattr(backbone, "num_prefix_tokens", 0) >= 1:
            blocks = list(backbone.blocks)
            dimensions = [backbone.num_features] * len(blocks)
        else:
            raise TypeError("Use a timm ViT with a class token or a Swin transformer.")
        if prompt_layers == "all":
            prompt_layers = list(range(1, len(blocks) + 1))
        self.prompt_layers = tuple(prompt_layers)
        if not self.prompt_layers or list(self.prompt_layers) != sorted(set(self.prompt_layers)):
            raise ValueError("prompt_layers must be a sorted nonempty list of one-based layer indices.")
        if min(self.prompt_layers) < 1 or max(self.prompt_layers) > len(blocks):
            raise ValueError("A prompted layer is outside the backbone depth.")
        if self.is_swin and (features != "cls" or lora or adapter or ssf):
            raise ValueError("features=cls_gap, lora, adapter, and ssf require a ViT backbone.")
        self.lora = self.adapters = self.ssf = None
        if ssf:
            width = backbone.num_features
            self.ssf = nn.ModuleDict({"patch_embed": ScaleShift(width), "norm": ScaleShift(width)})
            backbone.patch_embed.register_forward_hook(_transform_output(self.ssf["patch_embed"]))
            backbone.norm.register_forward_hook(_transform_output(self.ssf["norm"]))
            for layer, block in enumerate(blocks, 1):
                for target, find in SSF_TARGETS.items():
                    module = find(block)
                    channels = module.out_features if isinstance(module, nn.Linear) else module.normalized_shape[0]
                    self.ssf[f"{layer}_{target}"] = ScaleShift(channels)
                    module.register_forward_hook(_transform_output(self.ssf[f"{layer}_{target}"]))
        if lora:
            rank, targets = int(lora.get("rank", 8)), list(lora.get("targets", ["qkv"]))
            if rank < 1 or not targets or set(targets) - set(LORA_TARGETS):
                raise ValueError(f"lora needs a positive rank and targets from {sorted(LORA_TARGETS)}.")
            self.lora = nn.ModuleDict()
            for layer, block in enumerate(blocks, 1):
                for target in targets:
                    linear = LORA_TARGETS[target](block)
                    module = LowRank(linear.in_features, linear.out_features, rank,
                                     float(lora.get("alpha", rank)), float(lora.get("dropout", 0.0)))
                    self.lora[f"{layer}_{target}"] = module
                    linear.register_forward_hook(_add_to_output(module))
        if adapter:
            self.adapters = nn.ModuleDict()
            for layer, block in enumerate(blocks, 1):
                module = Adapter(backbone.num_features, int(adapter.get("dim", 32)), float(adapter.get("scale", 0.1)))
                self.adapters[str(layer)] = module
                block.mlp.register_forward_hook(_add_to_output(module))
        self.prompts = nn.ParameterDict({str(i): nn.Parameter(torch.empty(prompt_length, dimensions[i - 1]))
                                        for i in self.prompt_layers})
        for layer, prompt in zip(self.prompt_layers, self.prompts.values()):
            if prompt_init == "xavier":
                patch = getattr(getattr(backbone, "patch_embed", None), "patch_size", (16, 16))
                bound = math.sqrt(6.0 / (3 * patch[0] * patch[1] + dimensions[layer - 1]))
                nn.init.uniform_(prompt, -bound, bound)
            else:
                nn.init.normal_(prompt, std=prompt_std)
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.features = features
        self.head = nn.Linear(backbone.num_features * (2 if features == "cls_gap" else 1), num_classes)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _prompt(self, layer, batch):
        if str(layer) not in self.prompts:
            return None
        return self.prompt_dropout(self.prompts[str(layer)][None].expand(batch, -1, -1))

    def forward(self, images, return_states=False):
        states = []
        backbone = self.backbone
        if self.is_swin:
            x = backbone.patch_embed(images)
            layer = 0
            for stage in backbone.layers:
                x = stage.downsample(x)
                for block in stage.blocks:
                    layer += 1
                    prompt = self._prompt(layer, images.shape[0])
                    x = prompted_swin_block(block, x, prompt)
                    if return_states and prompt is not None:
                        states.append(LayerState(layer, x.flatten(1, 2), x.mean((1, 2))))
            final_pooled = x.mean((1, 2))
            features = backbone.norm(x).mean((1, 2))
        else:
            x = backbone.norm_pre(backbone.patch_drop(backbone._pos_embed(backbone.patch_embed(images))))
            prefix = backbone.num_prefix_tokens
            for layer, block in enumerate(backbone.blocks, 1):
                prompt = self._prompt(layer, images.shape[0])
                if prompt is not None:
                    x = torch.cat((x[:, :prefix], prompt, x[:, prefix:]), dim=1)
                x = block(x)
                if prompt is not None:
                    x = torch.cat((x[:, :prefix], x[:, prefix + prompt.shape[1]:]), dim=1)
                    if return_states:
                        states.append(LayerState(layer, x, x[:, 0]))
            final_pooled = x[:, 0]
            tokens = backbone.norm(x)
            features = backbone.fc_norm(tokens[:, 0])
            if self.features == "cls_gap":
                features = torch.cat((features, tokens[:, prefix:].mean(1)), dim=-1)
        logits = self.head(features)
        return {"logits": logits, "states": states, "final_pooled": final_pooled} if return_states else logits


def model_state(model, include_backbone=True):
    state = model.state_dict()
    if include_backbone:
        return state
    return {key: value for key, value in state.items() if not key.startswith("backbone.")}


def load_model_state(model, state, backbone_in_checkpoint=True):
    if backbone_in_checkpoint:
        model.load_state_dict(state)
        return
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("backbone.") for key in missing):
        raise ValueError(f"Checkpoint does not match the model: missing={missing[:8]}, unexpected={unexpected[:8]}")


def load_backbone_checkpoint(backbone, path):
    if Path(path).suffix.lower() in {".npz", ".npy"}:
        if not hasattr(backbone, "load_pretrained"):
            raise ValueError("This backbone does not support NumPy pretrained checkpoints.")
        backbone.load_pretrained(str(path))
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if any(key.startswith("base_encoder.") for key in state):
        state = {key.removeprefix("base_encoder."): value for key, value in state.items()
                 if key.startswith("base_encoder.")}
    expected = backbone.state_dict()
    filtered = {key: value for key, value in state.items() if key in expected}
    missing = sorted(set(expected) - set(filtered))
    if missing:
        raise ValueError(f"Backbone checkpoint is incomplete; missing keys: {missing[:8]}")
    backbone.load_state_dict(filtered, strict=True)


def create_model(config, pretrained=None):
    import timm

    options = dict(config["model"])
    name = options.pop("backbone", "vit_base_patch16_224.orig_in21k")
    settings = options.pop("backbone_kwargs", {})
    configured_pretrained = bool(options.pop("pretrained", True))
    use_pretrained = configured_pretrained if pretrained is None else pretrained
    checkpoint = options.pop("backbone_checkpoint", None)
    if checkpoint and use_pretrained and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    backbone = timm.create_model(name, num_classes=0, pretrained=use_pretrained and not checkpoint, **settings)
    if checkpoint and use_pretrained:
        load_backbone_checkpoint(backbone, checkpoint)
    return PromptedVisionTransformer(backbone, **options)
