"""
Loads and runs levels.

Based on and adapted from:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]
"""
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from models.VAE.vae import VAEMario
from models.GAN.gan import GANMario
from models.DCGAN.dcgan import DCGANMario
from mario_utils.levels import tensor_to_sim_level, clean_level

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

Tensor = torch.Tensor


filepath = Path(__file__).parent.resolve()
JARFILE_PATH = f"{filepath}/simulator.jar"


def test_level_from_decoded_tensor(
    level: Tensor,
    human_player: bool = False,
    max_time: int = 30,
    visualize: bool = True,
) -> dict:
    if len(level.shape) < 4:
        level = level.view(1, *level.shape)
    level = tensor_to_sim_level(level)[0]
    level = str(level)

    return run_level(
        level, human_player=human_player, max_time=max_time, visualize=visualize
    )


def test_level_from_int_tensor(
    level: Tensor,
    human_player: bool = False,
    max_time: int = 45,
    visualize: bool = True,
) -> dict:
    level = clean_level(level.detach().numpy())
    level = str(level)

    return run_level(
        level, human_player=human_player, visualize=visualize, max_time=max_time
    )


def test_level_from_int_array(
    level: np.ndarray,
    human_player: bool = False,
    max_time: int = 45,
    visualize: bool = False,
) -> dict:
    level = clean_level(level)
    level = str(level)

    return run_level(
        level, human_player=human_player, max_time=max_time, visualize=visualize
    )


def run_level(
    level: str,
    human_player: bool = False,
    max_time: int = 30,
    visualize: bool = False,
) -> dict:
    # Run the simulator.jar file with the given level
    if human_player:
        java = subprocess.Popen(
            ["java", "-cp", JARFILE_PATH, "geometry.PlayLevel", level],
            stdout=subprocess.PIPE,
        )
    else:
        java = subprocess.Popen(
            [
                "java",
                "-cp",
                JARFILE_PATH,
                "geometry.EvalLevel",
                level,
                str(max_time),
                str(visualize).lower(),
            ],
            stdout=subprocess.PIPE,
        )

    lines = java.stdout.readlines()
    res = lines[-1]
    res = json.loads(res.decode("utf8"))
    res["level"] = level

    return res


def test_level_from_z(z: Tensor, vae: VAEMario, human_player: bool = False) -> dict:
    """
    Passes the level that z generates
    through the simulator and returns
    a dict with results_playabilit_out_of_five.

    These results_playabilit_out_of_five are defined in
    simulator.jar <- EvaluationInfo.
    """
    # Get the level from the VAE
    res = vae.decode(z.view(1, -1)).probs.argmax(dim=-1)
    level = res[0]

    return test_level_from_decoded_tensor(level, human_player=human_player)


def load_levels_from_csv(path):
    df = pd.read_csv(path)
    level_data_list = []

    for index, row in df.iterrows():
        level_data_str = row['level']
        level_data = json.loads(level_data_str)
        level_data_list.append(level_data)

    return level_data_list

def play_levels_from_csv(csv_path):
    ROOT_DIR = Path(__file__).parent.resolve()

    level_data_list = load_levels_from_csv(csv_path)

    for index, level_data in enumerate(level_data_list):
        level_tensor = torch.tensor(level_data, dtype=torch.int)

        test_level_from_int_tensor(
            level_tensor, human_player=True, visualize=True
        )
        print(f"Level {index} played")

def play_and_evaluate_level(level_tensor, num_playthroughs=5):
    playability_results = []
    for _ in range(num_playthroughs):
        result = test_level_from_int_tensor(level_tensor, human_player=False, visualize=True)
        playability_results.append(1 if result["marioStatus"] == 1 else 0)
    return playability_results

if __name__ == "__main__":

    #eval_csv_path = 'models/VAE/vae_csv/num_lvl_100_normal.csv'        #vae_MAPElites.csv or vae_CMAES.csv or vae_normal.csv
    #eval_csv_path = 'models/DCGAN/dcgan_csv/dcgan_normal.csv'          #dcgan_MAPElites.csv or dcgan_GA.csv or dcgan_normal.csv
    eval_csv_path = 'models/MarioGPT/converted_levels_mariogpt.csv'

    output_dir = 'results_load_run_level'

    # just playing a level from csv via latent vectors
    csv_path = 'models/VAE/vae_csv/vae_normal.csv'
    play_levels_from_csv(csv_path)

    df = pd.read_csv(eval_csv_path)

    # Evaluate and save playability results
    playability_results = []

    for index, row in df.iterrows():
        level_id = row.iloc[0]
        level_data = row['level']
        level_data = json.loads(level_data)
        level_tensor = torch.tensor(level_data, dtype=torch.int)

        playability_counts = play_and_evaluate_level(level_tensor, num_playthroughs=1)
        playability_results.extend([(level_id, playability) for playability in playability_counts])

        # Create a df with playability results
    playability_df = pd.DataFrame(playability_results, columns=['level', 'playable'])

    # Save playability results to CSV
    playability_df.to_csv('playability_results.csv', index=False)

    # Plotting playability
    plt.bar(df.index, df['marioStatus'], color=['green' if s == 1 else 'red' for s in df['marioStatus']])
    plt.xlabel('Level')
    plt.ylabel('Playable (1) / Non-playable (0)')
    plt.title('Playability')
    plt.xticks(df.index, df['level'], rotation=90)
    plt.show()

