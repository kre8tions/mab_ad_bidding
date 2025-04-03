import numpy as np

class EpsilonGreedyMAB:
    def __init__(self, n_slots, epsilon=0.1):
        """Initialize the MAB with number of slots and epsilon."""
        self.n_slots = n_slots
        self.epsilon = epsilon
        self.counts = np.zeros(n_slots)  # Number of pulls per slot
        self.values = np.zeros(n_slots)  # Average reward per slot

    def select_slot(self):
        """Choose a slot: explore with prob epsilon, else exploit."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_slots)  # Random slot
        return np.argmax(self.values)  # Best-known slot

    def update(self, slot, reward):
        """Update counts and values based on reward."""
        self.counts[slot] += 1
        n = self.counts[slot]
        current_value = self.values[slot]
        self.values[slot] += (reward - current_value) / n

if __name__ == "__main__":
    # Test the algorithm
    mab = EpsilonGreedyMAB(n_slots=4, epsilon=0.1)
    print("Initial values:", mab.values)
    print("Initial counts:", mab.counts)
    slot = mab.select_slot()
    print("Selected slot:", slot)
    mab.update(slot, 1)  # Simulate a click
    print("Updated values:", mab.values)
    print("Updated counts:", mab.counts)