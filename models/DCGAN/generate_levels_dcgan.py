"""
Simulator to run simulator.jar (Mario Framework) using generated levels from trained DCGAN models.
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
import subprocess
import json
import sys
from pathlib import Path
from random import choices, random

import cma
import pandas as pd
import torch
import numpy as np
import os
import ast

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from tqdm import trange

from dcgan import DCGANMario
from mario_utils.levels import tensor_to_sim_level, clean_level

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
    res['coin'] = (level_arr == 10).sum().item()

    return res


def test_level_from_z(z: Tensor, dcgan: DCGANMario, human_player: bool = False) -> dict:
    """
    Passes the level that z generates
    through the simulator and returns
    a dict with results_playabilit_out_of_five.

    These results_playabilit_out_of_five are defined in
    simulator.jar <- EvaluationInfo.
    """
    print('z: Tensor')
    print(z)

    res = dcgan.decode(z.view(1, -1)).permute(0, 2, 3, 1) #

    fake_samples = res.view(-1, dcgan.height, dcgan.width, dcgan.n_sprites)
    level = fake_samples.argmax(dim=-1)

    return test_level_from_decoded_tensor(level, human_player=human_player, visualize=True)


def create_objective_function(dcgan: DCGANMario, device):
    def objective_function(z):
        # Decode the latent vector z and obtain the distribution
        res = dcgan.decode(torch.tensor(z).float().view(1, -1).to(device)).permute(0, 2, 3, 1)

        level = res.argmax(dim=-1)

        fitness_dic = test_level_from_decoded_tensor(level, human_player=False, visualize=False)

        level_factor = fitness_dic['lengthOfLevelPassedCells'] / 30 #30/30 --> level passed

        # Fitness F1 inspired by Volz et al. paper.
        # Creates obstacles but also enemies since jumps are needed in both cases.
        fitnesses = np.array([
                level_factor if level_factor < 1.0 else level_factor + fitness_dic['jumpActionsPerformed']
        ])

        # # Adapted from F1 but in regard to the enhancement of the numer of enemies,
        # # since they need jumps and can cause a game over, increasing difficulty.
        #fitnesses = np.array([
        #        level_factor if level_factor < 1.0 else level_factor + fitness_dic['numberOfEnemies']
        #])

        # # Fitness F1 inspired by Volz et al. paper.
        # # Creates easy levels.
        # # fitnesses = np.array([
        #        (level_factor - 60) if level_factor < 1.0 else (level_factor - fitness_dic['jumpActionsPerformed'])
        #])

        # Fitness return
        # -fitness for GA
        # -fitness only for CMA-ES
        # positive fitness for MapElites
        fitness = fitnesses.sum(), fitness_dic['jumpActionsPerformed']

        #print('Fitness:', fitness)
        #for f in fitnesses:
            #print(f)

        return fitness
    return objective_function


def create_via_genetic_algorithm(model, device, generations, generator_state_dict):
    model.generator.load_state_dict(generator_state_dict)  # Load the generator's state
    fitness = create_objective_function(model, device)

    population_size = 14  # lambda

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

    return test_level_from_z(torch.tensor(best_individual).float(), model, human_player=True)


def create_via_cmaes(model, device, cmaes_iter, population_size, generator_state_dict):
    model.generator.load_state_dict(generator_state_dict)  # Load the generator's state
    fitness = create_objective_function(model, device)

    population_size = population_size
    # increases variance of opportunities for exploration in latent space.
    start = np.random.uniform(-2, 2, (2,))
    es = cma.CMAEvolutionStrategy(start, 1.5, {'popsize': population_size,
                                                'verb_filenameprefix': './dat/'})
    es.optimize(fitness, iterations=cmaes_iter)

    file_path = 'dat/generator_0.0001_2500_5to1_mario_dcgan_epoch_1400_v2_cmaes_10_lvl_20_iter_14_pop_4.0.txt'
    if not os.path.exists(file_path):
        open(file_path, 'a').close()

    with open(file_path, 'a') as file:
        file.write(str(es.result) + '\n')

    print(es.result)
    cma_vector = torch.tensor(es.result[0]).float()

    return test_level_from_z(cma_vector, model, human_player=True)



def create_via_mapelites(model, device, total_iter, generator_state_dict):
    model.generator.load_state_dict(generator_state_dict)  # Load the generator's state
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
    model = DCGANMario(device="cuda" if torch.cuda.is_available() else "cpu")
    generator_state_dict = torch.load("trained_dcgan/generator/generator_0.0001_2500_5to1_mario_dcgan_epoch_1400.pt")

    num_levels = 1
    cmaes_iter = 20         # CMA-ES
    population_size = 14    # CMA-ES
    generations = 100       # GA
    itrs = 20               # Map-Elites

    outdir = './dcgan_csv'

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

        # #
        # # Use random z vector to generate level
        # #
        # # file_name = f'dcgan_num_levels_{num_levels}_normal.csv'
        # random_z = 3.0 * torch.randn((2,)).to(model.device)
        # results = test_level_from_z(random_z, model, human_player=False)

        # #
        # # Find z vector with CMA-ES
        # #
        # file_name = f'dcgan_num_levels_{num_levels}_cmaes_{cmaes_iter}_pop_{population_size}.csv'
        # results = create_via_cmaes(model, model.device, cmaes_iter, population_size,
        #                            generator_state_dict=generator_state_dict)

        # #
        # # Find z vector with GA
        # #
        # file_name = f'dcgan_num_levels_{num_levels}_GA_generations_{generations}.csv'
        # results= create_via_genetic_algorithm(model, model.device, generations=generations,
        #                                        generator_state_dict=generator_state_dict)


        #
        # Find z vector with MapElites
        #
        file_name = f'dcgan_num_levels_{num_levels}_mapelites_iter_{itrs}_emi5_v_x.csv'
        results = create_via_mapelites(model, model.device, total_iter=itrs, generator_state_dict=generator_state_dict)

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
