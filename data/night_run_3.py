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
codes = [G81Code.RotatedPlanarG81Code(d) for d in [19, 21, 23, 25, 27, 29]]
error_model = BiasedDepolarizingErrorModel(bias = 30, axis='Z')
decoder = G81RMPSDecoder.RotatedPlanarG81RMPSDecoder(chi=8) 

# Define the range for the physical error to simulate for
error_probability_min = 0.5
error_probability_max = 0.5
number_of_steps = 1

error_probabilities = np.linspace(error_probability_min, error_probability_max, number_of_steps)

# Define maximum number of simulation runs for each specific probability and code
max_runs = 10000


# Print run parameters prior to run
print('Codes:', [code.label for code in codes])
print('Error model:', error_model.label)
print('Decoder:', decoder.label)
print('Error probabilities:', error_probabilities)
print('Maximum runs:', max_runs)


# Run simulations and store the result as a list of dictionaries
data = [app.run(code, error_model, decoder, error_probability, max_runs=max_runs) 
        for code in codes for error_probability in error_probabilities]


### Save data to file in json format, named with the current datetime

time_now = str(datetime.now())

#with open(f"simulation_{time_now}_data.json", "w", encoding="utf-8") as file:
with open(f"night_run_3.json", "w", encoding="utf-8") as file:
    for entry in data:
        file.write(json.dumps(entry))
        file.write("\n")











