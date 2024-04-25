### Initial imports
import json
import matplotlib.pyplot as plt



### Load data

file_name_1 = "night_run_1.json"
file_name_2 = "night_run_2.json"
file_name_3 = "d_at_spec_point_30eta_run.json"
data = []

with open(file_name_1, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line.removesuffix('\n')))

with open(file_name_2, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line.removesuffix('\n')))

with open(file_name_3, "r", encoding="utf-8") as file:
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


fig = plt.figure(1, figsize=(8, 6))
plt.title('Simulering med G81-ytkoden, Z-bias = 30', fontsize=14)
plt.xlabel('Fysisk felsannolikhet, p', fontsize=12)
plt.ylabel('Logisk felsannolikhet', fontsize=12)
plt.xlim(error_probability_min-0.05, error_probability_max+0.05)
plt.ylim(-0.05, 0.65)

# add data
for code, xys in xy_map.items():
    legend_label = f"{code[31:]}"
    plt.plot(*zip(*xys), 'x-', label=legend_label)
plt.legend(loc='upper left', fontsize=11)
plt.grid()
plt.savefig("eta30_d_3to17_p_zerotospecialpoint_svenska.png")







