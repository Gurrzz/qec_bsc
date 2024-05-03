### Initial imports
import json
import matplotlib.pyplot as plt
import numpy as np
import locale

### Analytical solution

def h_g81(x):
    numerator = (12*x**2 + 1)**2
    denominator = (8*x**3 + 6*x)**2
    return numerator / denominator

def t_g81(x):
    return (h_g81(x) + 0.5) / (h_g81(x) + 1)


def h_XZZX(x):
    numerator = (12*x**2 + 1)
    denominator = (8*x**3 + 6*x)
    return numerator / denominator

def t_XZZX(x):
    return (h_XZZX(x) + 0.5) / (h_XZZX(x) + 1)

eta_values = np.linspace(1, 1000, 5000)
probability_values_g81 = t_g81(eta_values)
probability_values_XZZX = t_XZZX(eta_values)



# Use the above in the plot below

### Load data

file_name_1 = "eta_run_3.json"
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

locale.setlocale(locale.LC_NUMERIC, "sv_SE.utf8")
plt.rcParams['axes.formatter.use_locale'] = True

fig = plt.figure(1, figsize=(8, 6))
plt.title('Simulering och analytisk lösning med ytkod G81, $d = 3$, i särskilda punkten', fontsize=14)
plt.xlabel('Z-bias [$\eta$]', fontsize=12)
plt.ylabel('Logisk felsannolikhet', fontsize=12)
plt.xlim(eta_min-5, 500+5)
plt.ylim(0.45, 0.75)

# add data
for code, xys in xy_map.items():
    legend_label = f"{code[15:18]} simulerad"
    
    plt.plot(*zip(*xys), label=legend_label)

plt.plot(eta_values, probability_values_g81, label="$G81$ analytisk")
plt.plot(eta_values, probability_values_XZZX, c="r", label="$XZZX$ analytisk")
plt.legend(loc='upper right', fontsize=11)

plt.grid()
plt.savefig("var_eta_d_3_p_specialpoint_svenska_3_4.png")







