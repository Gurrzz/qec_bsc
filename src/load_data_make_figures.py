### Initial imports
import json
import matplotlib.pyplot as plt









### Load data

file_name = "example_data.json"
data = []

with open(file_name, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line.removesuffix('\n')))


### Perform analysis and plots

# recover the number of runs per probability, error model, and decoder (assume the same for all)
runs_per_prob = data[0]['n_run']
error_model = data[0]['error_model']
decoder = data[0]['decoder']


# prepare xy_map for plotting each code performance, with x = physical error rate and y = logical error rate. 

xy_map = {}
for run in data:
    xys = xy_map.setdefault(run['code'], [])
    xys.append((run['physical_error_rate'], run['logical_failure_rate']))

# format plot
error_probability_min = 0  # Kan också hämtas ur data för plottning, se label "error_probability"
error_probability_max = 0.5


fig = plt.figure(1, figsize=(12, 9))
plt.title('Rotated planar code simulation\n({} runs per probability, {} error model, {} decoder)'.format(runs_per_prob, error_model, decoder))
plt.xlabel('Physical error rate')
plt.ylabel('Logical failure rate')
plt.xlim(error_probability_min-0.05, error_probability_max+0.05)
plt.ylim(-0.05, 0.65)
# add data
for code, xys in xy_map.items():
    plt.plot(*zip(*xys), 'x-', label='{} code'.format(code))
plt.legend(loc='upper left')
plt.savefig("test.png")







