'''
Plotting Diversity and Similarity Scores of different approaches
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Diversity_Score_Results.csv')

sns.set(style="whitegrid")
color_palette = sns.color_palette('colorblind')

# Create a bar plot
plt.figure(figsize=(12, 8))
sns.barplot(data=df, x="Approach", y="Diversity Score", palette=color_palette, ci=None, width=0.3)
#y = "Avg Similarity"
#hue="Optimisation Method" #if used with CMAES etc.

plt.xlabel('Approach')
plt.ylabel('Diversity Score') # Average Similarity
plt.title('Quality Scores') # Average Similarity Scores

# Adjust y-axis limits for better visibility
plt.ylim(0.95, 1.0)

plt.xticks(rotation=45)
#plt.legend() # only with optim
plt.tight_layout()
plt.show()
