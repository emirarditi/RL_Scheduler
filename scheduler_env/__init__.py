from gymnasium.envs.registration import register

register(
     id="scheduler_env/Scheduler-v0",
     entry_point="scheduler_env.envs:SchedulerEnvironment",
     max_episode_steps=300,
)