# src/simulate.py
import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_ad_slots, simulate_click
from mab_algorithm import EpsilonGreedyMAB

def run_simulation(n_trials=1000, epsilon=0.3, seed=40):
    ad_slots = generate_ad_slots()
    ctrs = list(ad_slots.values())
    mab = EpsilonGreedyMAB(n_slots=len(ctrs), epsilon=epsilon)
    rewards = []
    choices = []
    slot_rewards = [0] * len(ctrs)
    rng_select = np.random.RandomState(seed)
    rng_reward = np.random.RandomState(seed)

    for t in range(n_trials):
        slot = mab.select_slot(rng=rng_select)
        reward = simulate_click(ctrs[slot], rng=rng_reward)
        mab.update(slot, reward)
        rewards.append(reward)
        choices.append(int(slot))
        slot_rewards[slot] += reward
        if t < 20:
            print(f"Trial {t}: Slot={slot}, Reward={reward}, Random={rng_select.random():.4f}, Values={mab.values}")

    return rewards, choices, ad_slots, mab, slot_rewards

if __name__ == "__main__":
    # Run simulation for multiple epsilon values
    epsilon_values = [0.1, 0.3, 0.5]
    for epsilon in epsilon_values:
        print(f"\nRunning simulation with epsilon={epsilon}")
        rewards, choices, ad_slots, mab, slot_rewards = run_simulation(n_trials=1000, epsilon=epsilon)
        slot_names = list(ad_slots.keys())
        ctrs = list(ad_slots.values())
        
        # Step 9: Analyze Results
        cumulative_reward = np.cumsum(rewards)
        optimal_reward = max(ctrs) * np.arange(1, 1001)
        regret = optimal_reward - cumulative_reward

        # Step 10: Visualize
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_reward, label="Cumulative Reward")
        plt.plot(optimal_reward, label="Optimal Reward", linestyle="--")
        plt.xlabel("Trial")
        plt.ylabel("Clicks")
        plt.title(f"Cumulative Reward Over Time (epsilon={epsilon})")
        plt.legend()
        plt.savefig(f"../visuals/cumulative_reward_epsilon_{epsilon}.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        slot_counts = [sum(1 for choice in choices if choice == i) for i in range(len(slot_names))]
        plt.bar(range(len(slot_names)), slot_counts, width=0.8, align="center")
        plt.xticks(ticks=range(len(slot_names)), labels=slot_names)
        plt.xlabel("Ad Slot")
        plt.ylabel("Number of Pulls")
        plt.title(f"Slot Selection Frequency (epsilon={epsilon})")
        plt.savefig(f"../visuals/slot_frequency_epsilon_{epsilon}.png")
        plt.close()

        # Print results
        print("Total clicks:", sum(rewards))
        print("Slot choices (first 10):", choices[:10])
        print("Final average rewards per slot:")
        for i, (name, reward, count, tot_reward) in enumerate(zip(slot_names, mab.values, mab.counts, slot_rewards)):
            print(f"  {name}: {reward:.4f} (True CTR: {ad_slots[name]}, Pulls: {count}, Clicks: {tot_reward})")
        print("Cumulative reward (last 5):", cumulative_reward[-5:])
        print("Regret (last 5):", regret[-5:])
        print("Last 5 rewards:", rewards[-5:])