# Imports

import numpy as np
import collections
import itertools
from datetime import datetime
import json

from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel, BiasedDepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.rotatedplanar import RotatedPlanarRMPSDecoder

from qecsim import app

import RotatedPlanarG81Code as G81Code
import RotatedPlanarG81Pauli as G81Pauli
import RotatedPlanarG81RMPSDecoder as G81RMPSDecoder


### Preparing simulations 

# Surface codes, error model and decoder type to use
code = G81Code.RotatedPlanarG81Code(3)
eta_min, eta_max = (1, 1000) # powers of 10
eta_vec = np.linspace(eta_min, eta_max, num=1000)
error_model = BiasedDepolarizingErrorModel(bias = 30, axis='Z')
decoder = G81RMPSDecoder.RotatedPlanarG81RMPSDecoder(chi=8) 

# Define the range for the physical error to simulate for

#error_probabilities = np.linspace(error_probability_min, error_probability_max, number_of_steps)

# Define maximum number of simulation runs for each specific probability and code
max_runs = 80000


# Print run parameters prior to run
#print('Codes:', [code.label for code in codes])
print('Code:', code.label)
print('Error model: (eta : 1-1000)', error_model.label)
print('Decoder:', decoder.label)
print('Error probabilities: (p_s)')
print('Maximum runs:', max_runs)


# Run simulations and store the result as a list of dictionaries

data = []
for eta in eta_vec:
    error_model = BiasedDepolarizingErrorModel(bias=eta, axis='Z')
    error_probability = (1 + 1/eta) / (2 + 1/eta)
    data.append(app.run(code, error_model, decoder, error_probability, max_runs=max_runs))

#data = [app.run(code, error_model, decoder, error_probability, max_runs=max_runs) 
#        for code in codes for error_probability in error_probabilities]


### Save data to file in json format, named with the current datetime

time_now = str(datetime.now())

#with open(f"simulation_{time_now}_data.json", "w", encoding="utf-8") as file:
with open(f"eta_run_3.json", "w", encoding="utf-8") as file:
    for entry in data:
        file.write(json.dumps(entry))
        file.write("\n")











