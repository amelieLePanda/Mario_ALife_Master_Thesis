'''
Counting amount of wins played out of 5.
'''
import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'results_playabilit_out_of_five', 'results_mariogpt.csv')

win_counts = defaultdict(int)

with open(file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)

    for row in csvreader:
        level_id = int(row[0])
        outcome = int(row[1])

        if outcome == 1:
            win_counts[level_id] += 1

win_rate_counts = Counter(win_counts.values())

# Num of Levels only loosing
lost_completely = sum([count for wins, count in win_rate_counts.items() if wins == 0])
win_rate_counts[0] = lost_completely

win_rate_counts = Counter(win_counts.values())

not_won_levels = 100 - len(win_counts)
win_rate_counts[0] = not_won_levels


for wins, count in win_rate_counts.items():
    print(f'Number of levels with {wins} wins:', count)

# Plotting
sns.set(style='whitegrid')
#greyscale_palette = ["#000000", "#333333", "#666666", "#999999", "#CCCCCC"]
#sns.set_palette('colorblind')
marine = "#0077B6"

# Create a bar plot using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x=list(win_rate_counts.keys()), y=list(win_rate_counts.values()), color=marine)
plt.xlabel('Number of Wins')
plt.ylabel('Number of Levels')
plt.title('Number of Levels by Win Count - Trained MarioGPT')
#plt.yticks(range(0, max(win_rate_counts.values()) + 1))

plt.show()
