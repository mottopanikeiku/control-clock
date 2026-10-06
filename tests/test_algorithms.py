import numpy as np
import torch

from control_clock.ppo import advantages_reference
from control_clock.search import ars_update, greedy_actions


def test_gae_matches_numpy_reference_with_truncation():
    rng = np.random.default_rng(7)
    shape = (9, 4)
    rewards = rng.normal(size=shape).astype(np.float32)
    values = rng.normal(size=shape).astype(np.float32)
    next_values = rng.normal(size=shape).astype(np.float32)
    terminated = rng.random(shape) < 0.2
    done = terminated | (rng.random(shape) < 0.2)
    expected = np.zeros(shape, dtype=np.float32)
    for slot in range(shape[1]):
        carry = 0.0
        for index in reversed(range(shape[0])):
            delta = rewards[index, slot] - values[index, slot]
            if not terminated[index, slot]:
                delta += 0.99 * next_values[index, slot]
            carry = delta + (0 if done[index, slot] else 0.99 * 0.95 * carry)
            expected[index, slot] = carry
    actual = advantages_reference(
        *[torch.from_numpy(array) for array in (rewards, values, next_values, terminated, done)],
        0.99,
        0.95,
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)


def test_ars_update_matches_loop_reference():
    rng = np.random.default_rng(19)
    weights = rng.normal(size=(3, 7))
    directions = rng.normal(size=(8, 3, 7))
    positive, negative = rng.normal(size=(2, 8))
    selected = np.argsort(np.maximum(positive, negative))[-4:]
    scale = np.std(np.concatenate((positive[selected], negative[selected])))
    expected = weights.copy()
    for index in selected:
        expected += 0.2 / (4 * scale) * (positive[index] - negative[index]) * directions[index]
    np.testing.assert_allclose(
        ars_update(weights, directions, positive, negative, 4, 0.2), expected, atol=1e-14
    )
    np.testing.assert_array_equal(
        ars_update(weights, directions, np.ones(8), np.ones(8), 4, 0.2), weights
    )


def test_affine_policy_matches_scalar_reference():
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(10, 3, 7))
    observations = rng.normal(size=(10, 6))
    scales = np.array([1, 1, 1, 1, 4 * np.pi, 9 * np.pi])
    expected = [np.argmax(w @ np.append(obs / scales, 1)) for w, obs in zip(weights, observations)]
    np.testing.assert_array_equal(greedy_actions(weights, observations, scales), expected)
    np.testing.assert_array_equal(
        greedy_actions(np.zeros((3, 7)), observations, scales), np.zeros(10)
    )
