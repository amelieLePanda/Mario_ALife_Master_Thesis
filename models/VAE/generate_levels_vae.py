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
from random import random, randint, choices, uniform

import cma
import torch
import numpy as np
from vae import VAEMario
import pandas as pd
import ast
from mario_utils.levels import tensor_to_sim_level, clean_level

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

import sys
from tqdm import tqdm, trange

torch.manual_seed(42)

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
    res['coin'] = (level_arr == 10).sum().item() # when using other training set
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
    res = vae.decode(z.view(1, -1)).probs.argmax(dim=-1) # a Categorical distribution
    level = res[0]

    return test_level_from_decoded_tensor(level, human_player=human_player, visualize=True)

def create_objective_function(vae: VAEMario, device):
    '''
    Different fitness functions, depending on objective.
    Use -fitness return for CMA-ES and Genetic Algorithm.
    Use positive fitness return for MapElites.
    '''
    def objective_function(z):
        # Decode the latent vector z and obtain the categorical distribution
        res = vae.decode(torch.tensor(z).float().view(1, -1).to(device))

        # Get the logits from the Categorical distribution
        #logits = res.logits
        #level = res.sample() # Categorical dist
        level = res.mode

        fitness_dic = test_level_from_decoded_tensor(level, human_player=False, visualize=False)

        level_factor = fitness_dic['lengthOfLevelPassedCells'] / 30 #30/30 --> level passed

        # Fitness F1 inspired by Volz et al. paper.
        # Creates obstacles but also enemies since jumps are needed in both cases.
        fitnesses = np.array([
                level_factor if level_factor < 1.0 else level_factor + fitness_dic['jumpActionsPerformed']
        ])

        # # Adapted from F1 but in regard to the enhancement of the numer of enemies,
        # # since they need jumps and can cause a game over, increasing difficulty.
        # fitnesses = np.array([
        #         level_factor if level_factor < 1.0 else level_factor + fitness_dic['numberOfEnemies']
        # ])

        # # Fitness F2
        # # Creates easy levels.
        # fitnesses = np.array([
        #         (level_factor - 60) if level_factor < 1.0 else (level_factor - fitness_dic['jumpActionsPerformed'])
        # ])

        # Fitness return
        # -fitness for GA
        # -fitness only for CMA-ES
        # positive fitness for MapElites
        fitness = -fitnesses.sum(), fitness_dic['jumpActionsPerformed']

        def combined_fitness(level_factor, fitness_dic):
            '''
            Combined fitness for enemies and jumpActionsPerformed.
            '''
            weight_ene = 1.0
            weight_obs = 1.0

            ene_fit= np.array([level_factor if level_factor < 1.0 else level_factor + fitness_dic['numberOfEnemies']])
            obs_fit = np.array([
                (level_factor - 60) if level_factor < 1.0 else (level_factor - fitness_dic['jumpActionsPerformed'])
        ])
            combined_fitness_value = (weight_ene * ene_fit) + (weight_obs * obs_fit)

            return combined_fitness_value
        
        fitness = fitnesses.sum()

        #print('Fitness:', fitness)
        #for f in fitnesses:
            #print(f)

        # Fitness return
        # -fitness for GA
        # -fitness only for CMA-ES
        # positive fitness for MapElites
        return -fitness, fitness_dic['jumpActionsPerformed']
    return objective_function

def create_via_cmaes(model, device, cmaes_iter, population_size):
    model.load_state_dict(torch.load("./trained_vae/0.0001_64_mario_vae_epoch_499_v3_StudenT.pt",
                                     map_location=device))
    fitness = create_objective_function(model, device)

    population_size = population_size  # lambda
    # increases variance of opportunities for exploration in latent space.
    start = np.random.uniform(-1, 1, (2,))
    es = cma.CMAEvolutionStrategy(start, 1.66, {'popsize': population_size,
                                                'verb_filenameprefix': './dat/'})
    es.optimize(fitness, iterations=cmaes_iter)

    file_path = 'dat/cmaes_test.txt'

    if not os.path.exists(file_path):
        open(file_path, 'a').close()

    with open(file_path, 'a') as file:
        file.write(str(es.result) + '\n')

    print(es.result)
    cma_vector = torch.tensor(es.result[0]).float()

    return test_level_from_z(cma_vector, model, human_player=False)

def  create_via_genetic_algorithm(model, device, generations):
    model.load_state_dict(
        torch.load("./trained_vae/0.0001_64_mario_vae_epoch_499_v3_StudenT.pt", map_location=device)) #0.0001_64_mario_vae_epoch_499_v3_StudenT.pt
    fitness = create_objective_function(model, device)

    population_size = 17  # lambda

    population = np.random.normal(0, 5/3, size=(population_size, 2))

    for generation in range(generations):
        # Evaluate fitness for each individual in the population
        fitness_scores = [fitness(individual) for individual in population]

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
    return test_level_from_z(torch.tensor(best_individual).float(), model, human_player=False)

def create_via_mapelites(model, device, total_iter):
    model.load_state_dict(torch.
                          load("./trained_vae/0.0001_64_mario_vae_epoch_499_v3_StudenT.pt", map_location=device))
    fitness = create_objective_function(model, device)

    solution_dim = 2

    archive = GridArchive(
        solution_dim=solution_dim,  # Dimensionality of solutions in archive.
        dims=[10],                  # 10 niches in one dimension.
        ranges=[(0, 20)],
        qd_score_offset=0,          # map elites is maximising
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            sigma0=1.0,
            x0=np.full(solution_dim, 0),  # Starting point for optimisation.
            ranker="2imp",
        ) for _ in range(5)
    ]

    scheduler = Scheduler(archive, emitters)

    for itr in trange(1, total_iter + 1, file=sys.stdout, desc='Iterations'):
        # Request trained_models from the scheduler.
        sols = scheduler.ask()

        results = [fitness(z) for z in sols]

        objs, meas = [], [] # objective function --> reward
        for obj, n_jumps in results:
            objs.append(obj)
            meas.append([n_jumps])

        # Send the results playability out of five back to the scheduler.
        scheduler.tell(objs, meas)

        elite = archive.retrieve_single([10])

    return test_level_from_z(torch.tensor(elite.solution).float(), model, human_player=False)

if __name__ == "__main__":
    model = VAEMario(device="cuda" if torch.cuda.is_available() else "cpu")
    model_name = "./trained_vae/0.0001_64_mario_vae_final_StudenT.pt"
    model.load_state_dict(torch.load(model_name))

    num_levels = 2
    cmaes_iter = 20             # CMA-ES
    population_size = 14        # CMA-ES
    generations = 100           # GA
    map_iter = 20               # Map-Elites

    outdir = './vae_csv'
    file_name = f'num_lvl_{num_levels}_normal_example_vae.csv'

    col_names = ['marioStatus', 'agentName', 'agentType', 'lengthOfLevelPassedPhys', 'totalLengthOfLevelPassedPhys',
                 'lengthOfLevelPassedCells', 'totalLengthOfLevelPassedCells', 'timeSpentOnLevel', 'totalTimeGiven',
                 'numberOfGainedCoins', 'totalActionsPerformed', 'jumpActionsPerformed', 'level',  'solid',
                 'breakableStone', 'empty', 'question', 'emptyQuestion', 'numberOfEnemies',
                 'leftPipeTop', 'rightPipeTop', 'leftPipe', 'rightPipe', 'coin']

    results_dict = {
        column_name: [] for column_name in col_names
    }

    # Training Loop, saving telemetries to csv
    for i in range(num_levels):
        # print("Starting run %i"%(i+1))
        human_player = False

        #
        # Use random z vector to generate level
        #
        random_z = 3.0 * torch.randn((2,)) # noise vector random z to generate levels from
        results = test_level_from_z(random_z, model, human_player=human_player)

        #
        # Find z vector with CMA-ES
        #
        #results = create_via_cmaes(model, model.device, cmaes_iter=cmaes_iter, population_size=population_size)

        #
        # Find z vector with MapElites
        #
        #results  = create_via_mapelites(model, model.device, total_iter=map_iter)

        #
        # Find z vector with GA
        #
        #results = create_via_genetic_algorithm(model, model.device, generations=generations)




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