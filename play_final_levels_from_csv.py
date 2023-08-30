"""
Simulator to run simulator.jar (Mario Framework) using generated levels from trained VAE models.
Playable as a human player or with Robin Baumgarten's super-human A* agent.

Based on and adapted from:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]

Map-Elites based and adapted from:
Bryon Tjanaka and Sam Sommerer and Nikitas Klapsis and Matthew C. Fontaine and Stefanos Nikolaidis, 2021.
Using CMA-ME to Land a Lunar Lander Like a Space Shuttle. Available from:
https://docs.pyribs.org/en/stable/tutorials/lunar_lander.html [Accessed 07.07.2023]

Fitness based on and adapted from:
Volz, V., Schrum, J., Liu, J., Lucas, S. M., Smith, A., & Risi, S.,  2018.
Evolving mario levels in the latent space of a deep convolutional generative adversarial network. In GECCO.
Aguirre, ed. Proceedings of the genetic and evolutionary computation conference, July 15 - 19, 2018, Kyoto Japan.
New York US: Association for Computing Machinery, pp. 221-228.
"""

import os
import subprocess
import json
from pathlib import Path
import torch
import numpy as np
from models.VAE.vae import VAEMario
import pandas as pd
import ast
from mario_utils.levels import tensor_to_sim_level, clean_level

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
        visualize: bool = False,
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
        visualize: bool = True,
) -> dict:
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
    # print(level)
    level_list = ast.literal_eval(level)
    level_arr = np.array(level_list, dtype=int)

    res['solid'] = (level_arr == 0).sum().item()  # stone
    res['breakableStone'] = (level_arr == 1).sum().item()
    res['empty'] = (level_arr == 2).sum().item()
    res['question'] = (level_arr == 3).sum().item()
    res['emptyQuestion'] = (level_arr == 4).sum().item()
    res["numberOfEnemies"] = (level_arr == 5).sum().item()
    res['leftPipeTop'] = (level_arr == 6).sum().item()
    res['rightPipeTop'] = (level_arr == 7).sum().item()
    res['leftPipe'] = (level_arr == 8).sum().item()
    res['rightPipe'] = (level_arr == 9).sum().item()
    res['coin'] = (level_arr == 10).sum().item()  # when using other training set
    return res


# For GAN: (z: Tensor, gan: GANMario, human_player: bool = False) -> dict:
def test_level_from_z(z: Tensor, vae: VAEMario, human_player: bool = False) -> dict:
    """
    Passes the level that z generates
    through the simulator and returns
    a dict with results_playabilit_out_of_five.

    These results_playabilit_out_of_five are defined in
    simulator.jar <- EvaluationInfo (marioaiDagstuhl/src/ch/idsia/tools/EvaluationInfo.java by Sergey Karakovskiy)
    """
    # Get the level from the VAE
    res = vae.decode(z.view(1, -1)).probs.argmax(dim=-1)  # a Categorical distribution
    level = res[0]

    return test_level_from_decoded_tensor(level, human_player=human_player)


if __name__ == "__main__":
    model = VAEMario(device="cuda" if torch.cuda.is_available() else "cpu")
    model_name = "./trained_vae/0.0001_64_mario_vae_epoch_499_v3_StudenT.pt"
    model.load_state_dict(torch.load(model_name))

    csv_path = 'models/VAE/vae_csv/num_lvl_5_0001_64_mario_vae_epoch_499_v3_StudenT_mapelites_10_3_teeeeeeeeest.csv'
    model_path = 'models/VAE/trained_vae/0.0001_64_mario_vae_epoch_499_v3_StudenT.pt'
    output_dir = 'models/VAE/vae_img'
    experiment_name = 'testrun'


    col_names = ['marioStatus', 'agentName', 'agentType', 'lengthOfLevelPassedPhys', 'totalLengthOfLevelPassedPhys',
                 'lengthOfLevelPassedCells', 'totalLengthOfLevelPassedCells', 'timeSpentOnLevel', 'totalTimeGiven',
                 'numberOfGainedCoins', 'totalActionsPerformed', 'jumpActionsPerformed', 'level', 'solid',
                 'breakableStone', 'empty', 'question', 'emptyQuestion', 'numberOfEnemies',
                 'leftPipeTop', 'rightPipeTop', 'leftPipe', 'rightPipe', 'coin']

    results_dict = {
        column_name: [] for column_name in col_names
    }

    # Training Loop, saving telemetries to csv
    for i in range(num_levels):
        # print("Starting run %i"%(i+1))
        human_player = False

        for column_name in col_names:
            if column_name == 'level':
                results_dict[column_name].append(ast.literal_eval(results[column_name]))
            else:
                results_dict[column_name].append(results[column_name])

        # print("Run %i done"%(i+1))

    df = pd.DataFrame(results_dict)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    dir_file = os.path.join(outdir, file_name)

    df.to_csv(dir_file)

    # df = pd.read_csv(file_name, header=0)
    # print(df['marioStatus'])

if __name__ == "__main__":
    csv_path = 'models/VAE/vae_csv/num_lvl_5_0001_64_mario_vae_epoch_499_v3_StudenT_mapelites_10_3_teeeeeeeeest.csv'
    model_path = 'models/VAE/trained_vae/0.0001_64_mario_vae_epoch_499_v3_StudenT.pt'
    output_dir = 'models/VAE/vae_img'
    experiment_name = 'testrun'

    play_levels_from_csv(csv_path, model_path, output_dir, experiment_name)
