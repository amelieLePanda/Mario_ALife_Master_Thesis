# Enhancing Artificial Intelligence Systems through ALife Principles and Generative Models: A Case Study with Super Mario Bros

![Image](https://github.com/amelieLePanda/Mario_ALife_Master_Thesis/blob/main/results_playabilit_out_of_five/difficult_mario_agent_fails_runs_back_multiple.gif)


**GAN, DCGAN, Minimal VAE** and **MarioGPT** can be located in the 'models' folder. 
Each folder includes the main scripts, such as:

* vae.py, vae_train, generate_levels_vae.py 
  

* gan.py, gan_train.py, generate_levels_gan.py


* dcgan.py, dcgan_train.py, generate_levels_dcgan.py

A random latent vector, CMAES, a simple genetic algorithm, or MAP-Elites, can be used to generate levels.


Each model has at least one pre-trained option available and can be found under trained_vae, trained_gan, and trained_dcgan.
To utilise the MarioGPT minimal implementation, please use:

- minimal_training_gpt.py 
- minimal_generation_gpt.py
- simulation_gpt.py 

**Note**: 
- MarioGPT was re-trained, and there were no changes to the architecture, only slight variations in the training settings. 
The minimal_generation_gpt.py file offers the option to utilise prompts, such as `prompts = ["many pipes, many enemies, some blocks, high elevation"]`.
- The pre-trained MarioGPT model can be found [here](https://drive.google.com/drive/folders/1KLLGjMD17G3N8SpJz2CU-ai4g_4DM3Iz?usp=sharing) (send a request for access)
- A sample of Gonzalez's original minimal VAE can be located at models/VAE/trained_vae/example.pt.


## Virtual Environment and Installing Requirements

Create a virtual environment and use at least Python version `>=3.9`. Then, install the necessary requirements by running this:

```
pip install -r requirements.txt
```


## Visualising Model

To visualise the latent space representation of a trained model, use:

```
python visualize.py
```


## Run Simulation

To execute the simulation using 'simulator.jar', a Java version above 8 (at least `OpenJDK 15.0.2`) is required.



## Play Level

To play a level yourself, use `human_player=True`, otherwise `human_player=False`. 
This will use Robin Baumgarten's A* agent instead.
Ensure that `visualize=True` is also set to true to view the simulation.

```
python load_run_level.py
```


## Bibliography and Software
Implementations are based, adapted and/or inspired by the following works:

Sudhakaran, s., Glanois, C., Freiberger, M., Najarro, E., & Risi S., 2023.
MarioGPT: Open-Ended Text2Level Generation through Large Language Models.
Available from: https://github.com/shyamsn97/mario-gpt/tree/main [Accessed 04 April 2023]

Paper: 
Sudhakaran, s., Glanois, C., Freiberger, M., Najarro, E., & Risi S., 2023.
MarioGPT: Open-Ended Text2Level Generation through Large Language Models. ArXiv [Online].
Available from: https://arxiv.org/abs/2302.05981 [Accessed 04 April 2023].

Code basis based on and adapted from:

González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]

González-Duque, M., Palm, R. B., Hauberg, S., & Risi, S., 2022. 
Mario plays on a manifold: Generating functional content in latent space through differential geometry. 
In 2022 IEEE Conference on Games (CoG) (pp. 385-392). IEEE.

DCGAN architecture inspired by:

Volz, V., Schrum, J., Liu, J., Lucas, S. M., Smith, A., & Risi, S.,  2018.
Evolving mario levels in the latent space of a deep convolutional generative adversarial network. In GECCO.
Aguirre, ed. Proceedings of the genetic and evolutionary computation conference, July 15 - 19, 2018, Kyoto Japan.
New York US: Association for Computing Machinery, pp. 221-228.

Student-t approach inspired by:

Mathieu, E., Rainforth, T., Siddharth, N., & Teh, Y. W., 2019.
Disentangling disentanglement in variational autoencoders.
In International conference on machine learning (pp. 4402-4412). PMLR.

Map-Elites based and adapted from:

Bryon Tjanaka and Sam Sommerer and Nikitas Klapsis and Matthew C. Fontaine and Stefanos Nikolaidis, 2021.
Using CMA-ME to Land a Lunar Lander Like a Space Shuttle. Available from:
https://docs.pyribs.org/en/stable/tutorials/lunar_lander.html [Accessed 07.07.2023]

VGLC:

Summerville, A. J., Snodgrass, S., Mateas, M., Onta'n Villar, S., 2016.
The VGLC: The Video Game Level Corpus
Proceedings of the 7th Workshop on Procedural Content Generation

Mario-AI-Framework:

Created by Ahmed Khalifa, based on work by Sergey Karakovskiy, Noor Shaker, and Julian Togelius, which in turn was based on Infinite Mario Bros by Markus Persson.
Available from: https://github.com/amidos2006/Mario-AI-Framework [Accessed 18.03.2023]

![](https://github.com/amelieLePanda/Mario_ALife_Master_Thesis/blob/main/results_playabilit_out_of_five/extreme_diff_mario_fail_high_quality.gif)
![Image](https://github.com/amelieLePanda/Mario_ALife_Master_Thesis/blob/main/results_playabilit_out_of_five/extreme_diff_mario_pass__high_quality.gif)
