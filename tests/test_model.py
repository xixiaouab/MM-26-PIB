import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from timm.models.swin_transformer import SwinTransformer

from pib.model import PromptedVisionTransformer, load_backbone_checkpoint, prompted_swin_block
from pib.losses import PIBObjective


def tiny_vit():
    return VisionTransformer(img_size=32, patch_size=8, embed_dim=24, depth=3,
                             num_heads=3, num_classes=0, mlp_ratio=2)


def test_frozen_backbone_keeps_prompt_gradients_and_removes_layer_prompts():
    model = PromptedVisionTransformer(tiny_vit(), 3, prompt_length=2).train()
    sizes = []
    hooks = [block.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[1]))
             for block in model.backbone.blocks]
    output = model(torch.randn(4, 3, 32, 32), return_states=True)
    for hook in hooks:
        hook.remove()
    assert sizes == [19, 19, 19]
    assert [state.tokens.shape[1] for state in output["states"]] == [17, 17, 17]
    assert not model.backbone.training
    objective = PIBObjective(3)
    objective(output, torch.tensor([0, 0, 1, 1]))["loss"].backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.grad.abs().sum() > 0 for parameter in model.prompts.values())
    assert model.head.weight.grad.abs().sum() > 0


def test_prompt_placement_and_inference_consistency():
    model = PromptedVisionTransformer(tiny_vit(), 3, prompt_layers=[1, 3], prompt_length=2).eval()
    x = torch.randn(2, 3, 32, 32)
    output = model(x, return_states=True)
    assert [state.index for state in output["states"]] == [1, 3]
    torch.testing.assert_close(output["logits"], model(x))
    with pytest.raises(ValueError):
        PromptedVisionTransformer(tiny_vit(), 3, prompt_layers=[3, 1])


def test_final_pooled_uses_last_backbone_block_after_last_prompt():
    model = PromptedVisionTransformer(tiny_vit(), 3, prompt_layers=[1], prompt_length=2).eval()
    final_block_output = []
    hook = model.backbone.blocks[-1].register_forward_hook(
        lambda module, args, output: final_block_output.append(output[:, 0]))
    output = model(torch.randn(2, 3, 32, 32), return_states=True)
    hook.remove()
    assert [state.index for state in output["states"]] == [1]
    torch.testing.assert_close(output["final_pooled"], final_block_output[0])
    assert not torch.allclose(output["final_pooled"], output["states"][-1].pooled)


def test_swin_window_prompts_and_hierarchical_state_dimensions():
    backbone = SwinTransformer(img_size=32, patch_size=4, embed_dim=12, depths=(2, 2),
                               num_heads=(3, 6), window_size=4, num_classes=0, mlp_ratio=2)
    model = PromptedVisionTransformer(backbone, 2, prompt_length=2).train()
    output = model(torch.randn(4, 3, 32, 32), return_states=True)
    assert [state.tokens.shape[-1] for state in output["states"]] == [12, 12, 24, 24]
    assert [state.tokens.shape[1] for state in output["states"]] == [64, 64, 16, 16]
    PIBObjective(4)(output, torch.tensor([0, 0, 1, 1]))["loss"].backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert all(p.grad.abs().sum() > 0 for p in model.prompts.values())


def test_swin_without_prompts_matches_original_shifted_block():
    backbone = SwinTransformer(img_size=32, patch_size=4, embed_dim=12, depths=(2,),
                               num_heads=(3,), window_size=4, num_classes=0, mlp_ratio=2).eval()
    block = backbone.layers[0].blocks[1]
    x = torch.randn(2, 8, 8, 12)
    torch.testing.assert_close(prompted_swin_block(block, x, None), block(x))


def test_backbone_checkpoint_loads_moco_prefix_and_rejects_missing_weights(tmp_path):
    backbone = tiny_vit()
    checkpoint = tmp_path / "moco.pt"
    state = {f"module.base_encoder.{key}": value for key, value in backbone.state_dict().items()}
    state["module.base_encoder.head.0.weight"] = torch.randn(4, 24)
    torch.save({"state_dict": state}, checkpoint)
    target = tiny_vit()
    load_backbone_checkpoint(target, checkpoint)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, backbone.state_dict()[key])
    torch.save({"model": {"cls_token": backbone.cls_token}}, checkpoint)
    with pytest.raises(ValueError, match="incomplete"):
        load_backbone_checkpoint(target, checkpoint)
