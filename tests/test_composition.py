import torch

from configs.config import CompositionConfig
from training.composition.model import build_model, parameter_groups
from training.composition.transforms import build_eval_transform, build_train_transform


def test_model_output_shape():
    model = build_model(num_classes=13, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 13)


def test_model_rejects_single_class():
    import pytest

    with pytest.raises(ValueError):
        build_model(num_classes=1, pretrained=False)


def test_parameter_groups_cover_all_params():
    model = build_model(num_classes=5, pretrained=False)
    groups = parameter_groups(model, lr_backbone=1e-4, lr_head=1e-3)
    n = sum(len(g["params"]) for g in groups)
    assert n == len(list(model.parameters()))
    assert groups[1]["lr"] == 1e-3 and len(groups[1]["params"]) == 2  # fc weight + bias


def test_transforms_produce_expected_shape(texture):
    pre = CompositionConfig().preprocessing()
    img = texture(1, size=(400, 300))
    for tf in (build_eval_transform(pre), build_train_transform(pre)):
        x = tf(img)
        assert x.shape == (3, pre["image_size"], pre["image_size"])


def test_eval_transform_is_deterministic(texture):
    pre = CompositionConfig().preprocessing()
    img = texture(2, size=(400, 300))
    tf = build_eval_transform(pre)
    assert torch.equal(tf(img), tf(img))


def test_config_preprocessing_is_serialisable():
    import json

    cfg = CompositionConfig()
    json.dumps(cfg.to_dict())
    assert cfg.splits_path.endswith("splits_group.csv")
