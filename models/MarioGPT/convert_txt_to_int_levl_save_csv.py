'''
Converts text representation files from the GPT model into a format compatible with
the simulation.jar utility of Minimal VAE.

Code structure based on and adapted from:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]
'''
import csv
import os

def change_txt_level_to_int(level_data):
    encoding = {
        "X": 0.0,
        "S": 1.0,
        "-": 2.0,
        "?": 3.0,
        "Q": 4.0,
        "E": 5.0,
        "<": 6.0,
        ">": 7.0,
        "[": 8.0,
        "]": 9.0,
        "o": 10.0,
        "x": 2.0,
        "b": 0.0,
        "B": 0.0,
    }

    converted_levels = []
    rows = level_data.strip().split('\n')

    for row in rows:
        converted_row = [encoding[symbol] for symbol in row]
        converted_levels.append(converted_row)

    return converted_levels


csv_file_path = "converted_levels_mariogpt.csv"
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["", "level"])

    # Iterate through files
    for i in range(100):
        print(i)
        input_file_path = os.path.join("txt_levels", f"generated_level_{i}.txt")

        with open(input_file_path, 'r') as f:
            level_data = f.read()

        converted_levels = change_txt_level_to_int(level_data)

        # Save CSV
        csv_writer.writerow([i, converted_levels])