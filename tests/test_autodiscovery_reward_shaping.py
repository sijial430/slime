from types import SimpleNamespace

import pytest

from slime.rollout.rm_hub.autodiscovery import post_process_rewards
from slime.utils.types import Sample


def _sample(reward: float, response_length: int) -> Sample:
    return Sample(index=0, reward=reward, response_length=response_length)


def test_length_control_preserves_raw_reward_and_shapes_advantage(monkeypatch):
    monkeypatch.setenv("AUTODISCOVERY_LENGTH_TARGET", "100")
    monkeypatch.setenv("AUTODISCOVERY_LENGTH_DEADBAND", "10")
    monkeypatch.setenv("AUTODISCOVERY_LENGTH_COEF", "1")
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
    )

    raw, advantages = post_process_rewards(args, [_sample(1.0, 100), _sample(1.0, 130)])

    assert raw == [1.0, 1.0]
    assert advantages == pytest.approx([0.1, -0.1])


def test_length_control_has_no_effect_inside_deadband(monkeypatch):
    monkeypatch.setenv("AUTODISCOVERY_LENGTH_TARGET", "100")
    monkeypatch.setenv("AUTODISCOVERY_LENGTH_DEADBAND", "10")
    monkeypatch.setenv("AUTODISCOVERY_LENGTH_COEF", "1")
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
    )

    raw, advantages = post_process_rewards(args, [_sample(0.0, 90), _sample(1.0, 110)])

    assert raw == [0.0, 1.0]
    assert advantages == pytest.approx([-0.5, 0.5])
