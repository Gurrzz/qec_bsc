### Initial imports
import json










### Load data

file_name = "example_data.json"
data = []

with open(file_name, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line.removesuffix('\n')))


### Perform analysis and plots

# recover the number of runs per probability (assume the same for all)
runs_per_prob = data[0]['n_run']

# prepare xy_map for plotting each code performance, with x = physical error rate and y = logical error rate. 

xy_map = {}








