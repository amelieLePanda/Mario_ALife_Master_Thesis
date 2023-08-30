'''
Runs level and lets an A* agent play it.

Original code by:
Sudhakaran, s., Glanois, C., Freiberger, M., Najarro, E., & Risi S., 2023.
MarioGPT: Open-Ended Text2Level Generation through Large Language Models.
Available from: https://github.com/shyamsn97/mario-gpt/tree/main [Accessed 04 April 2023]

Paper: Sudhakaran, s., Glanois, C., Freiberger, M., Najarro, E., & Risi S., 2023.
MarioGPT: Open-Ended Text2Level Generation through Large Language Models. ArXiv [Online].
Available from: URL [Accessed 04 April 2023].
'''

from mario_gpt import MarioLM

mario_lm = MarioLM()

prompts = ["many pipes, many enemies, some blocks, high elevation"]

generated_level = mario_lm.sample(
    prompts=prompts,
    num_steps=1400,
    temperature=2.0,
    use_tqdm=True
)

# play in interactive
generated_level.play()

# run Astar agent
generated_level.run_astar()