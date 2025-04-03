import numpy as np

# src/data_generator.py
import numpy as np

def generate_ad_slots():
    ad_slots = {"banner": 0.05, "sidebar": 0.02, "pop-up": 0.08, "footer": 0.03}
    return ad_slots

def simulate_click(slot_ctr, rng=None):
    return np.random.binomial(1, slot_ctr) if rng is None else rng.binomial(1, slot_ctr)

if __name__ == "__main__":
    slots = generate_ad_slots()
    print("Ad Slots:", slots)
    for slot, ctr in slots.items():
        # Run 1000 trials to estimate CTR
        clicks = [simulate_click(ctr) for _ in range(1000)]
        simulated_ctr = np.mean(clicks)
        print(f"{slot}: True CTR={ctr}, Simulated CTR={simulated_ctr:.4f}")


# if __name__ == "__main__":
#     slots = generate_ad_slots()
#     banner_clicks = [simulate_click(slots["banner"]) for _ in range(31)]
#     print("Banner clicks (31 trials):", banner_clicks)
#     print("Average:", np.mean(banner_clicks))