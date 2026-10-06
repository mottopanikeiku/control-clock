import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from control_clock.cleanrl_baseline import Args, make_agent
from control_clock.library_baselines import linear_schedule, load_settings


def test_zoo_task_settings_and_schedules():
    cart = load_settings("CartPole-v1")
    acrobot = load_settings("Acrobot-v1")
    lunar = load_settings("LunarLander-v3")
    assert (cart["n_envs"], cart["ppo"]["n_steps"], cart["ppo"]["batch_size"]) == (8, 32, 256)
    assert cart["ppo"]["n_epochs"] == 20
    assert cart["ppo"]["learning_rate"] == "lin_0.001"
    assert cart["vec_normalize"] is None
    assert acrobot["vec_normalize"]["norm_obs"]
    assert acrobot["vec_normalize"]["norm_reward"]
    assert acrobot["ppo"]["gae_lambda"] == 0.94
    assert lunar["ppo"]["gamma"] == 0.999
    assert lunar["ppo"]["ent_coef"] == 0.01
    assert linear_schedule(0.001)(0.5) == 0.0005


def test_sb3_evaluation_normalization_is_frozen():
    env = VecNormalize(DummyVecEnv([lambda: gym.make("Acrobot-v1")]))
    try:
        env.reset()
        count = env.obs_rms.count
        mean = env.obs_rms.mean.copy()
        variance = env.obs_rms.var.copy()
        observations = np.ones((100, 6), dtype=np.float32)
        normalized = env.normalize_obs(observations)
        assert normalized.shape == observations.shape
        assert env.obs_rms.count == count
        np.testing.assert_array_equal(env.obs_rms.mean, mean)
        np.testing.assert_array_equal(env.obs_rms.var, variance)
    finally:
        env.close()


def test_cleanrl_defaults_and_same_step_reset():
    args = Args()
    assert args.num_envs == 4 and args.num_steps == 128
    assert args.total_timesteps // 512 * 512 == 499712
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1", max_episode_steps=1)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        observations, _ = envs.reset(seed=[7])
        agent = make_agent(envs)
        with torch.no_grad():
            actions = agent.actor(torch.as_tensor(observations)).argmax(-1).numpy()
        _, rewards, _, truncated, infos = envs.step(actions)
        assert rewards[0] == 1
        assert truncated[0]
        assert "final_obs" in infos
        _, rewards, _, truncated, _ = envs.step(actions)
        # NEXT_STEP would instead ignore this action and return zero reward.
        assert rewards[0] == 1
        assert truncated[0]
    finally:
        envs.close()
