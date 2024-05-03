### Initial imports
import json
import numpy as np
import matplotlib.pyplot as plt
import locale


### Load data

file_name_1 = "d_at_spec_point_30eta_run.json"
file_name_2 = "d_at_spec_point_300eta_run.json"
file_name_3 = "d_at_spec_point_1000eta_run.json"
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
# runs_per_prob = data[0]['n_run']
# error_model = data[0]['error_model']
# decoder = data[0]['decoder']


# prepare xy_map for plotting each code performance, with x = physical error rate and y = logical error rate. 

xy_map = {}
for run in data:
    xys = xy_map.setdefault(run['error_model'], [])
    xys.append((run['n_k_d'][2], run['logical_failure_rate']))

# format plot
d_min = 3  # Kan också hämtas ur data för plottning, se label "error_probability"
d_max = 19

locale.setlocale(locale.LC_NUMERIC, "sv_SE.utf8")
plt.rcParams['axes.formatter.use_locale'] = True


fig = plt.figure(1, figsize=(8, 6))
plt.title('Simulering med ytkod G81 vid särskilda punkten för $p$', fontsize=14)
plt.xlabel('Kodavstånd $d$', fontsize=12)
plt.ylabel('Logisk felsannolikhet', fontsize=12)
plt.xlim(d_min-1, d_max+1)
plt.ylim(0.45, 0.6)

# add data
for error_model, xys in xy_map.items():
    legend_label = f"$\eta${error_model[25:-11]}"
    plt.plot(*zip(*xys), 'x-', label=legend_label)
plt.legend(loc='upper left', fontsize=11)
plt.xticks(np.arange(3,21,2))
plt.grid()
plt.savefig("var_eta_var_d_p_zerotospecialpoint_svenska.png")







