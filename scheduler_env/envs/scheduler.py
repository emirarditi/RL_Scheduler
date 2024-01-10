import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SchedulerEnvironment(gym.Env):
    def __init__(
            self, 
            no_of_task_types, 
            no_of_machines, 
            queue_size_min,
            queue_size_max, 
            task_machine_duration_map, 
            reward_type="makespan",
            include_energy_consumption=False,
            task_machine_energy_map=None,
            test=False,
            test_queue=None
            ):
        self.no_of_task_types = no_of_task_types
        self.no_of_machines = no_of_machines
        self.action_space = spaces.Discrete(no_of_machines)
        obs_space_size = no_of_task_types + no_of_task_types + no_of_task_types * no_of_machines
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_space_size,), dtype=np.float32)
        self.queue_size_min = queue_size_min
        self.queue_size_max = queue_size_max
        self.task_machine_duration_map = task_machine_duration_map
        self.tasks_queue = []
        self.task_type_counts = []
        self.machine_tasks = []
        self.reward_type = reward_type
        self.include_energy_consumption = include_energy_consumption
        self.task_machine_energy_map = task_machine_energy_map
        self.test = test
        self.test_queue = test_queue
        self.reset()

    def step(self, action):
        task_type = self.tasks_queue.pop(0) if len(self.tasks_queue) > 0 else None
        if task_type is not None:
            self.task_type_counts[task_type] -= 1
            self.machine_tasks[action][task_type] += 1
        terminated = False if len(self.tasks_queue) > 0 else True
        reward = 0
        if terminated:
            if self.reward_type == "makespan":
                # calculate Max time taken by any machine
                max_time_taken = 0
                for machine_id, machine in enumerate(self.machine_tasks):
                    time_taken = 0
                    for task_type, task_count in enumerate(machine):
                        time_taken += task_count * self.task_machine_duration_map[machine_id][task_type]
                    max_time_taken = max(max_time_taken, time_taken)
                reward = 1 / max_time_taken
            elif self.reward_type == "weighted_tardiness":
                for machine_id, machine in enumerate(self.machine_tasks):
                    for task_type, task_count in enumerate(machine):
                        reward -= task_count * self.task_machine_duration_map[machine_id][task_type]

            if self.include_energy_consumption:
                total_energy_consumption = 0
                for machine_id, machine in enumerate(self.machine_tasks):
                    for task_type, task_count in enumerate(machine):
                        total_energy_consumption += task_count * self.task_machine_energy_map[machine_id][task_type]
                reward += 1 / total_energy_consumption
            
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None, queue=None):
        super().reset(seed=seed)
        self.tasks_queue = []
        self.task_type_counts = [0 for _ in range(self.no_of_task_types)]
        self.machine_tasks = [[0 for __ in range(self.no_of_task_types)] for _ in range(self.no_of_machines)]
        # generate self.queue_size tasks randomly
        if self.test and self.test_queue is not None:
            queue = self.test_queue
        if queue is None:
            queue_size = self.np_random.integers(self.queue_size_min, self.queue_size_max)
            for _ in range(queue_size):
                task_type = self.np_random.integers(self.no_of_task_types)
                self.tasks_queue.append(task_type)
                self.task_type_counts[task_type] += 1
        else:
            for task_type in queue:
                self.tasks_queue.append(task_type)
                self.task_type_counts[task_type] += 1
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        obs = []
        curr_task = self.tasks_queue[0] if len(self.tasks_queue) > 0 else None
        task_hot_encoding = [0 for _ in range(self.no_of_task_types)]
        if curr_task is not None:
            task_hot_encoding[curr_task] = 1
        obs.extend(task_hot_encoding)
        obs.extend(self.task_type_counts)
        for machine in self.machine_tasks:
            obs.extend(machine)
        return obs
    

