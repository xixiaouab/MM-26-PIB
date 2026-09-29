from collections import Counter

import pytest
import torch

from pib.data import ClassBalancedBatchSampler


def test_balanced_batches_contain_distinct_repeated_class_samples():
    labels = [0] * 20 + [1] * 4 + [2] * 7 + [3] * 12
    sampler = ClassBalancedBatchSampler(labels, 8, 2, torch.Generator().manual_seed(10))
    for batch in sampler:
        counts = Counter(labels[index] for index in batch)
        assert len(counts) == 4
        assert set(counts.values()) == {2}
        assert len(batch) == len(set(batch))


def test_singleton_classes_do_not_duplicate_an_image():
    labels = [0] + [1] * 4 + [2] * 4
    sampler = ClassBalancedBatchSampler(labels, 6, 2, torch.Generator().manual_seed(1))
    for batch in sampler:
        counts = Counter(labels[index] for index in batch)
        assert counts == {0: 1, 1: 2, 2: 2}
        assert len(batch) == len(set(batch))


def test_repeated_classes_and_singletons_remain_trainable():
    labels = [0] * 4 + [1] * 4 + list(range(2, 20))
    sampler = ClassBalancedBatchSampler(labels, 4, 2, torch.Generator().manual_seed(1))
    batches = list(sampler)
    assert all(any(count >= 2 for count in Counter(labels[index] for index in batch).values())
               for batch in batches)
    assert any(labels[index] >= 2 for batch in batches for index in batch)


def test_sampler_reuses_loader_generator_for_resume_reproducibility():
    labels = [index // 5 for index in range(50)]
    generator = torch.Generator().manual_seed(8)
    sampler = ClassBalancedBatchSampler(labels, 8, generator=generator)
    list(sampler)
    checkpoint_state = generator.get_state()
    expected = list(sampler)
    fresh_generator = torch.Generator()
    fresh_generator.set_state(checkpoint_state)
    restored = ClassBalancedBatchSampler(labels, 8, generator=fresh_generator)
    assert list(restored) == expected


def test_balanced_batch_validation_and_small_class_count():
    with pytest.raises(ValueError, match="space"):
        ClassBalancedBatchSampler([0, 0, 1, 1], 2)
    with pytest.raises(ValueError, match="two training classes"):
        ClassBalancedBatchSampler([0] * 10, 4)
    labels = [0] * 8 + [1] * 8
    sampler = ClassBalancedBatchSampler(labels, 16)
    assert len(sampler) == 1
    assert Counter(labels[index] for index in next(iter(sampler))) == {0: 8, 1: 8}
