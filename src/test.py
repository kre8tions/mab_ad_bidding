import numpy as np
import matplotlib.pyplot as plt

print("NumPy version:", np.__version__)
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig("visuals/test_plot.png")
print("Test plot saved!")