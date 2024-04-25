### Initial imports
import json
import matplotlib.pyplot as plt



### Load data

file_name_1 = "eta_run_1.json"
data = []

with open(file_name_1, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line.removesuffix('\n')))

### Perform analysis and plots

# recover the number of runs per probability, error model, and decoder (assume the same for all)
runs_per_eta = data[0]['n_run']
decoder = data[0]['decoder']


# prepare xy_map for plotting each code performance, with x = physical error rate and y = logical error rate. 

xy_map = {}
for run in data:
    xys = xy_map.setdefault(run['code'], [])
    error_model = run['error_model']
    bias = float(error_model[26: -11])
    xys.append((bias, run['logical_failure_rate']))

# format plot
eta_min = 1  # Kan också hämtas ur data för plottning, se label "error_probability"
eta_max = 1000 


fig = plt.figure(1, figsize=(8, 6))
plt.title('Simulering med ytkod G81, d = 3', fontsize=14)
plt.xlabel('Z-bias', fontsize=12)
plt.ylabel('Logisk felsannolikhet', fontsize=12)
plt.xlim(eta_min-5, eta_max+5)
plt.ylim(0.45, 0.75)

# add data
for code, xys in xy_map.items():
    legend_label = f"{code[15:18]}, {code[31:]}"
    plt.plot(*zip(*xys), 'x-', label=legend_label)
# plt.legend(loc='upper right', fontsize=11)
plt.grid()
plt.savefig("var_eta_d_3_p_specialpoint_svenska.png")







