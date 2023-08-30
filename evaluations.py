'''
Figuring out similarity and diversity score of generated levels

Similarity Score and Diversity Score based and adapted from:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]
'''

import pandas as pd
import ast
import numpy as np

def read_levels_from_csv(file_path):
    df = pd.read_csv(file_path)
    df['level'] = df['level'].apply(lambda x: ast.literal_eval(x))
    levels = df['level'].tolist()

    return levels


def similarity(level1, level2):
    # Calculate the similarity between two levels
    return np.mean(level1 == level2)


def diversity(levels):
    # Calculate the diversity of a set of levels
    num_levels = len(levels)
    total_similarity = 0.0

    for i in range(num_levels):
        for j in range(i + 1, num_levels):
            total_similarity += similarity(levels[i], levels[j])

    avg_similarity = total_similarity / (num_levels * (num_levels - 1) / 2)
    diversity_score = 1 - 2 * avg_similarity

    return total_similarity, avg_similarity, diversity_score


if __name__ == "__main__":
    file_path = 'models/MarioGPT/gpt_csv/gpt_levels_int.csv'

    levels = read_levels_from_csv(file_path)

    # Calculate Similarity, Avg Similarity and Diversity Score
    total_similarity, avg_similarity, diversity_score = diversity(levels)

    print(f'Total Similarity: {total_similarity} Avg Similarity: {avg_similarity} Diversity Score: {diversity_score}')
