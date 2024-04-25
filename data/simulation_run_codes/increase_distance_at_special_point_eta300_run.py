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
codes = [G81Code.RotatedPlanarG81Code(d) for d in [3, 5, 7, 9, 11, 13, 15, 17, 19]]
error_model = BiasedDepolarizingErrorModel(bias = 300, axis='Z')
decoder = G81RMPSDecoder.RotatedPlanarG81RMPSDecoder(chi=8) 

# Define the range for the physical error to simulate for
# error_probability_min = 0
# error_probability_max = 0.5
# number_of_steps = 20

# error_probabilities = np.linspace(error_probability_min, error_probability_max, number_of_steps)
eta = 300
error_probability = (1 + 1/eta) / (2 + 1/eta)

# Define maximum number of simulation runs for each specific probability and code
max_runs = 10000


# Print run parameters prior to run
print('Codes:', [code.label for code in codes])
print('Error model:', error_model.label)
print('Decoder:', decoder.label)
print('Error probabilities:', error_probability)
print('Maximum runs:', max_runs)


# Run simulations and store the result as a list of dictionaries
data = [app.run(code, error_model, decoder, error_probability, max_runs=max_runs) 
        for code in codes]


### Save data to file in json format, named with the current datetime

time_now = str(datetime.now())

#with open(f"simulation_{time_now}_data.json", "w", encoding="utf-8") as file:
with open(f"d_at_spec_point_300eta_run.json", "w", encoding="utf-8") as file:
    for entry in data:
        file.write(json.dumps(entry))
        file.write("\n")











