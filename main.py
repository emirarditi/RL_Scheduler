import scheduler_env
import gymnasium as gym
from stable_baselines3 import DQN, PPO
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import itertools



NO_OF_MACHINES = 3
NO_OF_TASK_TYPES = 3
QUEUE_SIZE_MIN = 5
QUEUE_SIZE_MAX = 15
TOTAL_TIMESTEPS = 100000
REWARD_TYPE = "weighted_tardiness" # "makespan" or "weighted_tardiness"
INCLUDE_ENERGY_CONSUMPTION = True

RANDOM_QUEUE_TO_TEST = []

TASK_MACHINE_DURATION_MAP = {
    0: [1, 5, 1],
    1: [2, 10, 4],
    2: [5, 1, 4]
}

TASK_MACHINE_ENERGY_MAP = {
    0: [5, 4, 2],
    1: [3, 2, 1],
    2: [8, 6, 2]
}
queue_size_test = np.random.randint(QUEUE_SIZE_MIN, QUEUE_SIZE_MAX)
for _ in range(queue_size_test):
    RANDOM_QUEUE_TO_TEST.append(np.random.randint(NO_OF_TASK_TYPES))

def calculate_max_time_taken(actions, include_energy_consumption=INCLUDE_ENERGY_CONSUMPTION):
    machine_times = [0 for _ in range(NO_OF_MACHINES)]
    for task, machine in zip(RANDOM_QUEUE_TO_TEST, actions):
        machine_times[machine] += TASK_MACHINE_DURATION_MAP[int(machine)][task]
    rew = max(machine_times)
    if include_energy_consumption:
        energy_consumption = 0
        for task, machine in zip(RANDOM_QUEUE_TO_TEST, actions):
            energy_consumption += TASK_MACHINE_ENERGY_MAP[int(machine)][task]
        rew += 1 / energy_consumption
    return rew

def brute_force_solver():
    best_schedule = None
    best_completion_time = float('inf')
    # Generate all possible schedules
    all_schedules = itertools.product(range(NO_OF_MACHINES), repeat=queue_size_test)
    for schedule in all_schedules:
        completion_time = calculate_max_time_taken(schedule)
        if completion_time < best_completion_time:
            best_completion_time = completion_time
            best_schedule = schedule
    return best_schedule, best_completion_time

def make_env(test=False):
    return gym.make("scheduler_env/Scheduler-v0", 
                    no_of_task_types=NO_OF_TASK_TYPES, 
                    no_of_machines=NO_OF_MACHINES, 
                    queue_size_min=QUEUE_SIZE_MIN,
                    queue_size_max=QUEUE_SIZE_MAX,
                    task_machine_duration_map=TASK_MACHINE_DURATION_MAP,
                    include_energy_consumption=INCLUDE_ENERGY_CONSUMPTION,
                    task_machine_energy_map=TASK_MACHINE_ENERGY_MAP,
                    reward_type=REWARD_TYPE,
                    test=test,
                    test_queue=(RANDOM_QUEUE_TO_TEST if test else None)
    )

def random_solver():
    return np.random.randint(NO_OF_MACHINES, size=queue_size_test)

def rl_solver():
    env = make_env()
    test_env = make_env(test=True)


    model = PPO("MlpPolicy", env, verbose=0, stats_window_size=int(TOTAL_TIMESTEPS / ((QUEUE_SIZE_MIN + QUEUE_SIZE_MAX) / 2)))
    #stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=10, verbose=1)
    eval_callback = EvalCallback(test_env, log_path="./logs/", eval_freq=1000, deterministic=True, render=False)
    rewards_test = []
    rew_buffer = []
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10000, callback=eval_callback)

    with np.load('logs/evaluations.npz') as data:
        eval_data = data['results']
        for i in range(len(eval_data)):
            rewards_test.append(eval_data[i][0])

    # plot the test rewards
    plt.plot(rewards_test)
    plt.title(f"Test Queue Rewards across Training" + (" (with energy consumption)" if INCLUDE_ENERGY_CONSUMPTION else " (without energy consumption)") + f" (with {REWARD_TYPE} reward)")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.show()

    rew_buffer.extend([a["r"] for a in model.ep_info_buffer])
    # calculate average rewards for every 10 episodes
    avg_rew = []
    for i in range(0, len(rew_buffer), 10):
        avg_rew.append(np.mean(rew_buffer[i:i+10]))

    # plot the average rewards
    plt.plot(avg_rew)
    plt.title(f"Average Rewards across Training" + (" (with energy consumption)" if INCLUDE_ENERGY_CONSUMPTION else " (without energy consumption)") + f" (with {REWARD_TYPE} reward)")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.show()
    obs, _ = test_env.reset()
    done = False
    actions = []
    while not done:
        action, _states = model.predict(np.array(obs))
        actions.append(action)
        obs, rewards, done, _, info = test_env.step(action)
    return actions


rl_policy = rl_solver()
random_policy = random_solver()
brute_force_policy, _ = brute_force_solver()
print("Max time taken by RL Policy: ", calculate_max_time_taken(rl_policy))
print("Max time taken by Random Policy: ", calculate_max_time_taken(random_policy))
print("Max time taken by Brute Force Policy: ", calculate_max_time_taken(brute_force_policy))