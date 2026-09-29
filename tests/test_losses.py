import pytest
import torch
from torch.nn import functional as F

from pib.losses import PIBObjective, compression_loss, layer_schedule, sufficiency_loss
from pib.model import LayerState


@pytest.mark.parametrize("tokens,dim", [(5, 9), (9, 5)])
def test_compression_equals_explicit_off_diagonal_energy(tokens, dim):
    x = torch.randn(3, tokens, dim, requires_grad=True)
    normalized = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    correlation = normalized.transpose(-2, -1) @ normalized / tokens
    mask = ~torch.eye(dim, dtype=torch.bool)
    expected = correlation[:, mask].square().sum(-1).mean()
    actual = compression_loss(x)
    torch.testing.assert_close(actual, expected)
    grad_actual = torch.autograd.grad(actual, x, retain_graph=True)[0]
    grad_expected = torch.autograd.grad(expected, x)[0]
    torch.testing.assert_close(grad_actual, grad_expected, atol=1e-7, rtol=1e-4)


def test_compression_zero_for_orthogonal_dimensions():
    assert compression_loss(torch.eye(4)[None]).item() == 0
    assert compression_loss(torch.ones(1, 2, 2)).item() == pytest.approx(0.5)


def test_sufficiency_matches_class_scatter_equation():
    features = torch.tensor([[0., 0.], [2., 0.], [4., 0.], [6., 0.]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = sufficiency_loss(features, labels)
    torch.testing.assert_close(loss, torch.tensor(1 / 16))
    loss.backward()
    assert features.grad.abs().sum() > 0
    torch.testing.assert_close(sufficiency_loss(features.detach() * 3, labels), loss.detach())


def test_sufficiency_handles_singletons_and_one_class():
    features = torch.randn(3, 6, requires_grad=True)
    for labels in (torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0])):
        loss = sufficiency_loss(features, labels)
        assert loss.item() == 0
        assert loss.requires_grad
    mixed = torch.tensor([[0., 0.], [2., 0.], [5., 0.]])
    torch.testing.assert_close(sufficiency_loss(mixed, torch.tensor([0, 0, 1])), torch.tensor(1 / 16))


def test_path_penalizes_deterioration_and_routes_receive_gradients():
    states = [LayerState(1, torch.eye(2)[None].repeat(4, 1, 1), torch.randn(4, 2)),
              LayerState(2, torch.ones(4, 2, 2), torch.randn(4, 2))]
    labels = torch.tensor([0, 0, 1, 1])
    output = {"logits": torch.randn(4, 2, requires_grad=True), "states": states}
    loss = PIBObjective(2, compression=1., sufficiency=0., alpha=0., delta=.1,
                        path_weight=2., normalization="none", routing=True)
    values = loss(output, labels)
    torch.testing.assert_close(values["path"], torch.tensor(.4))
    torch.testing.assert_close(values["pib"], torch.tensor(.25))
    torch.testing.assert_close(values["loss"], F.cross_entropy(output["logits"], labels) + .25 + .8)
    values["loss"].backward()
    assert loss.routing_logits.grad[1] > 0


def test_normalization_preserves_gradients_and_warmup_zero_is_task_only():
    x = torch.randn(4, 7, 5, requires_grad=True)
    states = [LayerState(1, x, x[:, 0]), LayerState(2, x * 2 + 1, x[:, 0] + .2)]
    output = {"states": states, "logits": torch.randn(4, 2, requires_grad=True)}
    labels = torch.tensor([0, 0, 1, 1])
    objective = PIBObjective(2)
    values = objective(output, labels)
    values["loss"].backward()
    assert x.grad.abs().sum() > 0
    torch.testing.assert_close(objective(output, labels, 0)["loss"], F.cross_entropy(output["logits"], labels))


def test_schedules_support_depth_endpoints_and_validate_lengths():
    torch.testing.assert_close(layer_schedule({"start": 0, "end": 1}, 3), torch.tensor([0., .5, 1.]))
    with pytest.raises(ValueError):
        layer_schedule([1, 2], 3)
    with pytest.raises(ValueError):
        PIBObjective(2, delta=-.1)
