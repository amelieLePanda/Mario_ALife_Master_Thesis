"""
Saving levels as generated levels images, laoding data from csv, convert .txt files to level format

Based on and adapted from:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]
"""

import csv

import math
import matplotlib.pyplot as plt
import seaborn
from PIL import Image
import pandas as pd
import os
import json
from pathlib import Path

import torch
import numpy as np
import PIL

encoding = {
    "X": 0,
    "S": 1,
    "-": 2,
    "?": 3,
    "Q": 4,
    "E": 5,
    "<": 6,
    ">": 7,
    "[": 8,
    "]": 9,
    "o": 10,
    "x": 0,
    "Y": 0,
}

sprites = {
    encoding["X"]: "./mario_utils/sprites/stone.png",
    encoding["S"]: "./mario_utils/sprites/breakable_stone.png",
    encoding["?"]: "./mario_utils/sprites/question.png",
    encoding["Q"]: "./mario_utils/sprites/depleted_question.png",
    encoding["E"]: "./mario_utils/sprites/goomba.png",
    encoding["<"]: "./mario_utils/sprites/left_pipe_head.png",
    encoding[">"]: "./mario_utils/sprites/right_pipe_head.png",
    encoding["["]: "./mario_utils/sprites/left_pipe.png",
    encoding["]"]: "./mario_utils/sprites/right_pipe.png",
    encoding["o"]: "./mario_utils/sprites/coin.png",
}

def save_level_as_image(level_data, save_path, encoding, sprites):
    def get_img_from_level(level: np.ndarray) -> np.ndarray:
        image = []
        for row in level:
            image_row = []
            for c in row:
                if c == encoding["-"]:
                    # White background
                    tile = (255 * np.ones((16, 16, 3))).astype(int)
                elif c == -1:
                    # Masked
                    tile = (128 * np.ones((16, 16, 3))).astype(int)
                else:
                    sprite_path = sprites[c]
                    sprite_path = os.path.join(Path(__file__).parent.resolve(), sprite_path)
                    tile = np.asarray(PIL.Image.open(sprite_path).convert("RGB")).astype(int)
                image_row.append(tile)
            image.append(image_row)

        image = [np.hstack([tile for tile in row]) for row in image]
        image = np.vstack([np.asarray(row) for row in image])

        return image

    level = np.array(level_data)
    level_image = get_img_from_level(level)

    # Convert numpy array to PIL image
    level_image_pil = PIL.Image.fromarray(level_image.astype(np.uint8))

    level_image_pil.save(save_path)

def load_levels_from_csv_and_save_image(path, output_dir, experiment_name):
    df = pd.read_csv(path)

    for index, row in df.iterrows():
        level_data_str = row['level']
        level_data = json.loads(level_data_str)

        filename = f"level_{index}_{experiment_name}.png"
        save_path = os.path.join(output_dir, filename)
        save_level_as_image(level_data, save_path, encoding, sprites)

        print(f"Level {index} saved as {filename}")

def convert_txt_level(txt_path, output_dir, encoding):
    csv_filename = 'gpt_levels.csv'
    csv_file_path = os.path.join(output_dir, csv_filename)

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['level'])

    for filename in os.listdir(txt_path):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(txt_path, filename)

            with open(txt_file_path, 'r') as file:
                level_text = file.read()

            numerical_level = [[encoding[char] for char in row] for row in level_text.split('\n') if row.strip()]

            level_string = "[{}]".format(", ".join("[{}]".format(", ".join(map(str, row))) for row in numerical_level))

            with open(csv_file_path, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([level_string])

def resize_level():
    file_path = '/Master_Thesis_Generating_SMB/csv/_old_csv/gpt_levels.csv'

    # shorten GPT files to 14x15 files - not needed anymore
    sequences = []

    with open(file_path, "r") as csvfile:
        for row in csvfile.readlines():
            tokens = row.strip().split(',')
            sequences += list(map(lambda x: int(x), tokens))

    max_length = 14 * 100

    tensor_14x100 = torch.tensor(sequences, dtype=torch.int).reshape(14, 100)

    print(tensor_14x100)
    print(tensor_14x100.shape)

    # reshape to 14x15
    tensor_14x15 = tensor_14x100[:, :15]

    print(tensor_14x15)
    print(tensor_14x15.shape)

# Concatenating generated levels images from a run
def big_picture():
    folder_path = "./models/VAE/vae_img/big_pictures"
    output_path = "./models/VAE/vae_img/cmaes/big_picture.png"

    image_filenames = [filename for filename in os.listdir(folder_path) if filename.endswith(".png")]

    num_images = len(image_filenames)
    num_cols = 4

    num_rows = math.ceil(num_images / num_cols)

    if num_rows == 0:
        num_rows = 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image_path = os.path.join(folder_path, image_filenames[i])
            image = plt.imread(image_path)
            ax.imshow(image)
            ax.set_title(f"Level {i + 1}", pad=15)
        ax.axis("off")

    plt.tight_layout()

    # Save the combined image
    plt.savefig(output_path, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    path = 'models/MarioGPT/gpt_csv/gpt_levels_int.csv'     #models/DCGAN/dcgan_csv/dcgan_normal.csv'
    outdir = 'generated levels images/mariogpt'             #./models/DCGAN/dcgan_img/dcgan_normal'
    expname = 'mariogpt'

    #load_levels_from_csv_and_save_image(path, outdir, experiment_name=expname)

    load_levels_from_csv_and_save_image(path, outdir, expname)
    #big_picture()