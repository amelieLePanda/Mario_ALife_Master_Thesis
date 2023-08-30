"""
Simulator to run simulator.jar (Mario Framework) using generated levels from trained GAN models.
Playable as a human player or with Robin Baumgarten's super-human A* agent.

Structure based on:
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
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from random import random, choices
import cma
import ast
import sys
import time
from tqdm import tqdm, trange

from gan import GANMario
from mario_utils.levels import tensor_to_sim_level, clean_level

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler


Tensor = torch.Tensor

filepath = Path(__file__).parent.resolve()
JARFILE_PATH = f"{filepath}/simulator.jar"

def test_level_from_decoded_tensor(
    level: Tensor,
    human_player: bool = False,
    max_time: int = 30,
    visualize: bool = False,
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
    #print(level)

    level_list = ast.literal_eval(level)
    level_arr = np.array(level_list, dtype=int)

    res['solid'] = (level_arr == 0).sum().item() # stone
    res['breakableStone'] = (level_arr == 1).sum().item()
    res['empty'] = (level_arr == 2).sum().item()
    res['question'] = (level_arr == 3).sum().item()
    res['emptyQuestion'] = (level_arr == 4).sum().item()
    res["numberOfEnemies"] = (level_arr == 5).sum().item()
    res['leftPipeTop'] = (level_arr == 6).sum().item()
    res['rightPipeTop'] = (level_arr == 7).sum().item()
    res['leftPipe'] = (level_arr == 8).sum().item()
    res['rightPipe'] = (level_arr == 9).sum().item()
    res['coin'] = (level_arr == 10).sum().item()

    return res

def test_level_from_z(z: Tensor, gan: GANMario, human_player: bool = False) -> dict:
    """
    Passes the level that z generate through the simulator and returns
    a dict with results_playabilit_out_of_five. These results_playabilit_out_of_five are defined in:
    simulator.jar <- EvaluationInfo (marioaiDagstuhl/src/ch/idsia/tools/EvaluationInfo.java by Sergey Karakovskiy)
    """

    # Get the level from GAN
    res = gan.decode(z.view(1, -1))

    level = res[0].argmax(dim=-1)

    print(level)

    return test_level_from_decoded_tensor(level, human_player=human_player, visualize=True)

def create_objective_function(gan: GANMario, device):
    def objective_function(z):
        # Decode the latent vector z and obtain the categorical distribution
        res = gan.decode(torch.tensor(z).float().view(1, -1).to(device))

        # Get the logits from the Categorical distribution
        #logits = res.logits
        #level = res.sample() # Categorical dist

        level = torch.argmax(res, dim=-1)

        num_enemies = (level == 5).sum().item()
        left_pipe_top = (level == 6).sum().item()
        right_pipe_top = (level == 7).sum().item()
        solid = (level == 0).sum().item()

        #print('num_enemies')
        #print(num_enemies)

        fitness_dic = test_level_from_decoded_tensor(level, human_player=False, visualize=False)

        fitness_dic['leftPipeTop'] = left_pipe_top
        fitness_dic['rightPipeTop'] = right_pipe_top
        fitness_dic['numberOfEnemies'] = num_enemies
        fitness_dic['solid'] = solid

        #print('fitness_dic after level decoded')
        #print(fitness_dic)

        # # Weights based on importance
        # completion_weight = 5.0         # important
        # total_length_weight = 1.0       # getting as far as possible
        # time_penalty_weight = -0.05     # Penalty for time spent !
        # difficulty_weight = 1.0         # difficulty: jump actions, the more, the better
        # num_enemies_weight = 1.0/10     # difficulty: the more enemies, the better
        # jumps_weight = 1/28             # important as indicator for difficulty
        #
        # # coins_weight = 0              # not important regarding difficulty
        # # obstacles_weight = 1.0        # keep out, since kind of difficult to combine
        #                                 # obstacles could be placed where the agent can't even reach them.
        #                                 # Jump is already an indicator for obstacles in the agents way
        #
        # fitnesses = np.array([          # might have too many factors, hard to tune
        #         completion_weight * (fitness_dic['marioStatus']),
        #         total_length_weight * fitness_dic['lengthOfLevelPassedCells'] / 30, # in %
        #         time_penalty_weight * min(0, fitness_dic['timeSpentOnLevel'] - fitness_dic['totalTimeGiven']) +
        #         difficulty_weight * (1 /fitness_dic['totalActionsPerformed']) +
        #         num_enemies_weight * (fitness_dic['numberOfEnemies'])
        #         jumps_weight * fitness_dic['jumpActionsPerformed']
        # ])

        level_factor = fitness_dic['lengthOfLevelPassedCells'] / 30 #30/30 --> level passed

        fitnesses = np.array([ # F1 from paper, enhance jumps
                level_factor if level_factor < 1.0 else level_factor + fitness_dic['jumpActionsPerformed']
        ])
        #fitnesses = np.array([
        #        level_factor if level_factor < 1.0 else level_factor + fitness_dic['numberOfEnemies']
        #])
        '''fitnesses = np.array([ # F2 from paper --> try to make as easy as possible
                (level_factor - 60) if level_factor < 1.0 else (level_factor - fitness_dic['jumpActionsPerformed'])
        ])'''
        fitness = -fitnesses.sum()

        #print('Fitness:', fitness)
        #for f in fitnesses:
            #print(f)

        return fitness
    return objective_function


def create_via_cmaes(model, device, cmaes_iter):
    es = cma.CMAEvolutionStrategy(2 * [0], 1)
    fitness = create_objective_function(model, device)

    es.optimize(fitness, iterations=cmaes_iter)

    print(es.result)
    cma_vector = torch.tensor(es.result[0]).float().to(device)

    return test_level_from_z(cma_vector, model, human_player=True)

def  create_via_genetic_algorithm(model, device, generations):
    fitness = create_objective_function(model, device)
    population_size = 14  # lambda
    population = np.random.normal(0, 5/3, size=(population_size, 2))

    for generation in range(generations):
        # Evaluate fitness for each individual in the population
        fitness_scores = [fitness(individual) for individual in population]
        # fitness_scores = [fitness(model.encode(individual)) for individual in population]

        # Choose individuals with higher fitness scores
        selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=False)
        selected_population = [population[i] for i in selected_indices[:population_size//2]] # keep the good half

        # Reproduce
        offspring = selected_population.copy()
        for i in range(population_size//2): # only the better ones as parents
            parent1, parent2 = choices(selected_population, k=2)
            offspring.append(torch.tensor([(parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2]).float())

        # Mutation
        mutation_rate = 0.1
        for i in range(population_size):
            if random() < mutation_rate:
                offspring[i] += np.random.uniform(-0.1, 0.1, size=(2,))

        # Replace the old population with the new offspring
        print('add new offspring to population')
        population = offspring

    # Return the best individual from the final population
    best_individual = min(population, key=lambda ind: fitness(ind))

    print('fitness of best individual')
    print(fitness(best_individual))
    print([fitness(x) for x in population])
    return test_level_from_z(torch.tensor(best_individual).to(device).float(), model, human_player=False)

def create_via_mapelites(model, device, itrs):
    fitness = create_objective_function(model, device)

    solution_dim = 2

    archive = GridArchive(
        solution_dim=solution_dim,  # Dimensionality of solutions in the archive.
        dims=[10],                  # 10 niches along one dimension
        ranges=[(0, 20)],
        qd_score_offset=0,           # map elites is maximising
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            sigma0=1.0,
            x0=np.full(solution_dim, 0),  # Starting point for optimization
            ranker="2imp",
        ) for _ in range(5)
    ]

    scheduler = Scheduler(archive, emitters)

    total_itrs = itrs

    for itr in trange(1, total_itrs + 1, file=sys.stdout, desc='Iterations'):
        # Request trained_models from the scheduler.
        sols = scheduler.ask()

        results = [fitness(z) for z in sols]

        objs, meas = [], [] # objective function --> reward
        for obj, n_jumps in results:
            objs.append(obj)
            meas.append([n_jumps])

        # Send the results_playabilit_out_of_five back to the scheduler.
        scheduler.tell(objs, meas)

        elite = archive.retrieve_single([10])

    return test_level_from_z(torch.tensor(elite.solution).float(), model, human_player=False)


if __name__ == "__main__":
    generator_state_dict = torch.load("trained_gan/generator/generator0.0002_20000_mariogan_epoch_8500.pt")
    model = GANMario(device="cuda" if torch.cuda.is_available() else "cpu")
    model.generator.load_state_dict(generator_state_dict)  # Load the generator's state

    num_levels = 1
    cmaes_iter = 2          # CMA-ES
    population_size = 18    # CMA-ES
    generations = 2         # GA
    itrs = 5                # Map-Elites

    outdir = './gan_csv'

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

        # # Use random z vector to generate level
        file_name = f'gan_num_levels_{num_levels}_normal.csv'
        random_z = 3.0 * torch.randn((2,)).to(model.device)
        results = test_level_from_z(random_z, model, human_player=human_player)

        # Find z vector with CMA-ES
        #file_name = f'gan_num_levels_{num_levels}_cmaes_{cmaes_iter}.csv'
        #results_playabilit_out_of_five = create_via_cmaes(model, model.device, cmaes_iter)

        # # Find z vector with GA
        # file_name = f'gan_num_levels_{num_levels}_ga_{generations}.csv'
        # results_playabilit_out_of_five = create_via_genetic_algorithm(model, model.device, run_name='test', generations=generations)

        # # Find z vector with MapElites
        # file_name = f'gan_num_levels_{num_levels}_mape_{itrs}.csv'
        # results_playabilit_out_of_five = create_via_mapelites(model, model.device)

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