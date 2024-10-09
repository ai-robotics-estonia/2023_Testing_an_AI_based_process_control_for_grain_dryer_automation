"""
main.py

This script is part of the IntelliDry project, which aims to develop an AI-based process control system for grain dryer automation.

Usage:
  This script loads weekly dataset of a dryer and extract drying cycles

Author:
  Veiko Vunder <veiko.vunder@ut.ee>

Year:
  2014
"""

from dataset import Dataset
import matplotlib.pyplot as plt

# Load dataset
dataset = Dataset()
dataset.load_weekly_dataset_2024('W37_temperatures')
dataset.extract_drying_cycles()
cycles = dataset.extract_drying_cycles()

# Print drying cycles
print(f'Found {len(cycles)} drying cycles:')
for cycle in cycles:
  print(cycle)


# Plot drying cycles
dataset.plot_weekly_data(cycles)

plt.show()
