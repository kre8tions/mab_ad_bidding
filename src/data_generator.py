import numpy as np

def generate_ad_slots():
    """Define ad slots with fixed CTRs."""
    np.random.seed(42)  # For reproducibility
    ad_slots = {
        "banner": 0.05,  # 5% CTR
        "sidebar": 0.02, # 2% CTR
        "pop-up": 0.08,  # 8% CTR
        "footer": 0.03   # 3% CTR
    }
    return ad_slots

def simulate_click(slot_ctr):
    """Simulate a click based on CTR (Bernoulli trial)."""
    return np.random.binomial(1, slot_ctr)

if __name__ == "__main__":
    # Test the functions
    slots = generate_ad_slots()
    print("Ad Slots:", slots)
    for slot, ctr in slots.items():
        click = simulate_click(ctr)
        print(f"{slot}: CTR={ctr}, Click={click}")