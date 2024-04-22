# %matplotlib inline
import numpy as np
import collections
import itertools
import matplotlib.pyplot as plt

from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel, BiasedDepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.rotatedplanar import RotatedPlanarRMPSDecoder

from qecsim import app

import RotatedPlanarG81Code as G81Code
import RotatedPlanarG81Pauli as G81Pauli
import RotatedPlanarG81RMPSDecoder as G81RMPSDecoder


my_code = G81Code.RotatedPlanarG81Code(3)
my_error_model = DepolarizingErrorModel()
my_decoder = G81RMPSDecoder.RotatedPlanarG81RMPSDecoder(chi=8)

#print(app.run_once(my_code, my_error_model, my_decoder, 0.2))

#print(app.run_once(RotatedPlanarCode(3, 3), my_error_model, RotatedPlanarRMPSDecoder(chi=8), 0.2))


# Multiple runs 

codes = [G81Code.RotatedPlanarG81Code(d) for d in [3, 5, 7, 9]]
error_model = BiasedDepolarizingErrorModel(bias=30, axis='Z')
decoder = G81RMPSDecoder.RotatedPlanarG81RMPSDecoder(chi=8)

# Physical error
error_probability_min, error_probability_max = (0, 0.5)
error_probabilities = np.linspace(error_probability_min, error_probability_max, 10)

# Set max runs for each probability
max_runs = 1000

# Print run parameters
print('Codes:', [code.label for code in codes])
print('Error model:', error_model.label)
print('Decoder:', decoder.label)
print('Error probabilities:', error_probabilities)
print('Maximum runs:', max_runs)


# run simulations and print data from middle run to view format
data = [app.run(code, error_model, decoder, error_probability, max_runs=max_runs)
        for code in codes for error_probability in error_probabilities]
print(data[len(data)//2])


#
# prepare code to x,y map and print
code_to_xys = {}
for run in data:
    xys = code_to_xys.setdefault(run['code'], [])
    xys.append((run['physical_error_rate'], run['logical_failure_rate']))
print('\n'.join('{}: {}'.format(k, v) for k, v in code_to_xys.items()))


# format plot
fig = plt.figure(1, figsize=(12, 9))
plt.title('Rotated planar code simulation\n({} error model, {} decoder)'.format(error_model.label, decoder.label))
plt.xlabel('Physical error rate')
plt.ylabel('Logical failure rate')
plt.xlim(error_probability_min-0.05, error_probability_max+0.05)
plt.ylim(-0.05, 0.65)
# add data
for code, xys in code_to_xys.items():
    plt.plot(*zip(*xys), 'x-', label='{} code'.format(code))
plt.legend(loc='upper left')
plt.savefig("test.png")

