'''
Generates and plays level with trained MarioGPT model. Outputs img and level as txt file.

Based on and adapted from:
Sudhakaran, s., Glanois, C., Freiberger, M., Najarro, E., & Risi S., 2023.
MarioGPT: Open-Ended Text2Level Generation through Large Language Models.
Available from: https://github.com/shyamsn97/mario-gpt/tree/main [Accessed 04 April 2023]

Paper: Sudhakaran, s., Glanois, C., Freiberger, M., Najarro, E., & Risi S., 2023.
MarioGPT: Open-Ended Text2Level Generation through Large Language Models. ArXiv [Online].
Available from: URL [Accessed 04 April 2023].
'''

import torch
from mario_gpt import MarioLM, SampleOutput
from transformers import AutoModel
from huggingface_hub import notebook_login
# pretrained_model = shyamsn97/Mario-GPT2-700-context-length

#notebook_login()
#access_token = 'hf_mxxbJDXrInuyPpkyiTCbSrgXVWEcnqKBD' # via hugging-face, needs own access account
#model = AutoModel.from_pretrained('shyamsn97/Mario-GPT2-700-context-length', token=access_token)
#model = torch.load('./Mario-GPT2-700-context-length/iteration_99/pytorch_model.bin')

mario_lm = MarioLM()
mario_lm.load_pretrained_lm('./Mario-GPT2-700-context-length/iteration_99/', {})

device = torch.device('cuda')
mario_lm = mario_lm.to(device)

prompts = ["many pipes, many enemies, some blocks, high elevation"]

# Generate multiple levels
for i in range(0, 2):
    print('Start generating levels:')
    # generate level of size 210, pump temperature up to ~2.4 for more stochastic but playable levels
    generated_level = mario_lm.sample(
        prompts=prompts,
        num_steps=210,
        temperature=2.0,
        use_tqdm=True
    )
    # save image
    img_filename = f"generated_level_{i}.png"
    generated_level.img.save(img_filename)

    # save text level to file
    file_filename = f"generated_level_{i}.txt"
    generated_level.save(file_filename)

    # run Astar agent
    generated_level.run_astar()

print('Levels generated and saved as png and txt files.')

# show string list
#generated_level.level

# show PIL image
#generated_level.img

# play in interactive
# generated_level.play()

# Continue generation
# generated_level_continued = mario_lm.sample(
#     seed=generated_level,
#     prompts=prompts,
#     num_steps=1400,
#     temperature=2.0,
#     use_tqdm=True
# )

# load from text file
# loaded_level = SampleOutput.load("generated_level_2.txt")

# play from loaded (should be the same level that we generated)
# loaded_level.play()
