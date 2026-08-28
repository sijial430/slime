import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout.rm_hub import autodiscovery
from slime.rollout.rm_hub.autodiscovery import post_process_rewards
from slime.utils.types import Sample

NUM_GPUS = 0


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
        reward_key=None,
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
        reward_key=None,
    )

    raw, advantages = post_process_rewards(args, [_sample(0.0, 90), _sample(1.0, 110)])

    assert raw == [0.0, 1.0]
    assert advantages == pytest.approx([-0.5, 0.5])


def test_feedback_accumulates_only_reward_one_rollouts(tmp_path, monkeypatch):
    state = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(autodiscovery, "_FEEDBACK_STATE", str(state))

    async def summarize(analysis, error):
        return f"summary: {analysis}"

    monkeypatch.setattr(autodiscovery, "_summarize_learnings", summarize)

    def record(hypothesis, reward, rollout_id, analysis, dataset_id="archaeology"):
        asyncio.run(
            autodiscovery._record_feedback(
                dataset_id,
                hypothesis,
                reward,
                rollout_id=rollout_id,
                execution_log=f"\nAnalysis:\n{analysis}\nReview:\naccepted",
            )
        )

    record("successful hypothesis 1", 1.0, 0, "first result")
    unchanged = state.read_text()
    record("failed hypothesis", 0.0, 1, "failed result")
    assert state.read_text() == unchanged

    record("successful hypothesis 2", 1.0, 2, "second result")
    record("other hypothesis", 1.0, 2, "other result", dataset_id="another-task")

    entries = autodiscovery._successful_feedback("archaeology", k=0)
    assert [entry["hypothesis"] for entry in entries] == [
        "successful hypothesis 2",
        "successful hypothesis 1",
    ]
    assert autodiscovery._successful_feedback("archaeology", k=1) == entries[:1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
