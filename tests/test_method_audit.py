import pytest
import torch
from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer

from pib.losses import PIBObjective, compression_loss, scatter_statistics, sufficiency_loss
from pib.model import LayerState, PromptedVisionTransformer


def explicit_off_diagonal(tokens, eps=1e-8):
    energies = []
    for image in tokens.float():
        normalized = image / (image.norm(dim=-1, keepdim=True) + eps)
        correlation = normalized.transpose(0, 1) @ normalized / image.shape[0]
        mask = ~torch.eye(image.shape[1], dtype=torch.bool)
        energies.append(correlation[mask].square().sum())
    return torch.stack(energies).mean()


@pytest.mark.parametrize("num_tokens,dim", [(3, 8), (8, 3)])
def test_compression_matches_explicit_formula_and_gradient(num_tokens, dim):
    torch.manual_seed(112)
    tokens = torch.randn(2, num_tokens, dim, requires_grad=True)
    actual = compression_loss(tokens)
    expected = explicit_off_diagonal(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    actual_gradient = torch.autograd.grad(actual, tokens, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, tokens)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-5)


def test_compression_excludes_diagonal_and_averages_per_image():
    independent = torch.eye(2)
    correlated = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    assert compression_loss(independent[None]).item() == 0
    torch.testing.assert_close(compression_loss(correlated[None]), torch.tensor(0.32))
    torch.testing.assert_close(compression_loss(torch.stack((independent, correlated))), torch.tensor(0.16))


def test_sufficiency_averages_valid_classes_and_counts_ordered_class_pairs():
    features = torch.tensor([[0.0], [2.0], [4.0], [6.0], [12.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2])
    intra, inter, valid, classes = scatter_statistics(features, labels)
    torch.testing.assert_close(intra, torch.tensor(2.0))
    torch.testing.assert_close(inter, torch.tensor(372.0))
    assert (valid, classes) == (2, 3)
    loss = sufficiency_loss(features, labels)
    torch.testing.assert_close(loss, torch.tensor(2 / 372))
    loss.backward()
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("labels", [torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0])])
def test_sufficiency_handles_all_singletons_or_a_single_class(labels):
    features = torch.randn(3, 4, requires_grad=True)
    loss = sufficiency_loss(features, labels)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(features.grad, torch.zeros_like(features))


def test_path_hinge_penalizes_deterioration_and_preserves_feature_gradient():
    first = torch.eye(2).expand(2, -1, -1).clone().requires_grad_()
    second = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]).expand(2, -1, -1).clone().requires_grad_()
    output = {"logits": torch.zeros(2, 2), "states": [LayerState(1, first, torch.zeros(2, 2)), LayerState(2, second, torch.zeros(2, 2))]}
    objective = PIBObjective(2, compression=0, sufficiency=0, alpha=0, delta=0.1, path_weight=1, normalization="none", routing=False)
    result = objective(output, torch.tensor([0, 1]))
    torch.testing.assert_close(result["quality"], torch.tensor([0.0, 0.32]))
    torch.testing.assert_close(result["path"], torch.tensor(0.22))
    result["path"].backward()
    assert second.grad.abs().sum() > 0
    output["states"].reverse()
    assert objective(output, torch.tensor([0, 1]))["path"].item() == 0


def test_routing_gradient_is_sigmoid_weighted_compression_minus_sufficiency():
    tokens = torch.randn(4, 3, 5)
    pooled = torch.tensor([[0.0, 1.0], [1.0, 2.0], [4.0, 1.0], [6.0, 2.0]])
    output = {"logits": torch.zeros(4, 2), "states": [LayerState(1, tokens, pooled)]}
    labels = torch.tensor([0, 0, 1, 1])
    objective = PIBObjective(1, compression=2, sufficiency=3, path_weight=0, normalization="none", routing=True)
    result = objective(output, labels)
    expected = 0.25 * (2 * compression_loss(tokens) - 3 * sufficiency_loss(pooled, labels))
    result["pib"].backward()
    torch.testing.assert_close(objective.routing_logits.grad[0], expected)


@pytest.mark.parametrize("kind", ["vit", "swin"])
def test_information_losses_train_prompts_without_encoder_gradients(kind):
    torch.manual_seed(53)
    if kind == "vit":
        backbone = VisionTransformer(img_size=16, patch_size=8, embed_dim=24, depth=2, num_heads=3, num_classes=0)
    else:
        backbone = SwinTransformer(img_size=16, patch_size=4, embed_dim=16, depths=(2, 1), num_heads=(2, 4), window_size=2, num_classes=0)
    model = PromptedVisionTransformer(backbone, 2, prompt_length=2).train()
    assert not backbone.training
    output = model(torch.randn(4, 3, 16, 16), return_states=True)
    labels = torch.tensor([0, 0, 1, 1])
    objective = PIBObjective(len(model.prompt_layers), normalization="none")
    result = objective(output, labels)
    regularizer = result["pib"] + objective.path_weight * result["path"]
    regularizer.backward()
    assert all(prompt.grad is not None and prompt.grad.abs().sum() > 0 for prompt in model.prompts.values())
    assert objective.routing_logits.grad is not None
    assert not any(parameter.grad is not None for parameter in backbone.parameters())
    if kind == "vit":
        assert all(state.tokens.shape[1] == 5 for state in output["states"])
