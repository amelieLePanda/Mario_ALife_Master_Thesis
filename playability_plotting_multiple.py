'''
Counting amount of wins played out of 5, comparing multiple files.
'''

import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


def process_file(file_path, win_counts):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            level_id = int(row[0])
            outcome = int(row[1])

            if outcome == 1:
                win_counts[level_id] += 1


script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results_playabilit_out_of_five')

csv_files = ['results_vae_cmaes.csv', 'results_dcgan_cmaes.csv']

# Set the style and color palette
sns.set(style='whitegrid')
#greyscale_palette = ["#000000", "#333333", "#666666", "#999999", "#CCCCCC"]
sns.set_palette('colorblind')
colors = sns.color_palette()

plt.figure(figsize=(10, 6))

x_offset = 0

for idx, csv_file in enumerate(csv_files):
    win_counts = defaultdict(int)
    file_path = os.path.join(results_dir, csv_file)
    process_file(file_path, win_counts)

    win_rate_counts = Counter(win_counts.values())

    lost_completely = sum([count for wins, count in win_rate_counts.items() if wins == 0])
    win_rate_counts[0] = lost_completely

    not_won_levels = 40 - len(win_counts) # adapt to number of levels!
    win_rate_counts[0] = not_won_levels

    x_values = [wins + x_offset for wins in win_rate_counts.keys()]
    plt.bar(x_values, win_rate_counts.values(), color=colors[idx], width=0.2, label=os.path.splitext(csv_file)[0])

    x_offset += 0.2

plt.xlabel('Number of Wins')
plt.ylabel('Number of Levels')
plt.title('Win Counts VAE-CMAES and DCGAN-CMAES')
plt.legend()
plt.tight_layout()
plt.show()