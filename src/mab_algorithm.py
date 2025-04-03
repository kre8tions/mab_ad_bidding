# src/mab_algorithm.py
import numpy as np

class EpsilonGreedyMAB:
    def __init__(self, n_slots, epsilon=0.2):
        self.n_slots = n_slots
        self.epsilon = epsilon
        self.counts = np.zeros(n_slots)
        self.values = np.zeros(n_slots)

    def select_slot(self, rng=None):
        if np.all(self.values == 0):
            return np.random.randint(self.n_slots) if rng is None else rng.randint(self.n_slots)
        rand = np.random.random() if rng is None else rng.random()
        if rand < self.epsilon:
            return np.random.randint(self.n_slots) if rng is None else rng.randint(self.n_slots)
        return np.argmax(self.values)

    def update(self, slot, reward):
        self.counts[slot] += 1
        n = self.counts[slot]
        current_value = self.values[slot]
        self.values[slot] += (reward - current_value) / n