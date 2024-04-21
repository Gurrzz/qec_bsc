import numpy as np

from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.rotatedplanar import RotatedPlanarRMPSDecoder

from qecsim import app

import RotatedPlanarG81Code as G81Code
import RotatedPlanarG81Pauli as G81Pauli
import RotatedPlanarG81RMPSDecoder as G81RMPSDecoder


my_code = G81Code.RotatedPlanarG81Code(3)
my_error_model = DepolarizingErrorModel()
my_decoder = G81RMPSDecoder.RotatedPlanarG81RMPSDecoder(chi=8)

print(app.run_once(my_code, my_error_model, my_decoder, 0.2))

#print(app.run_once(RotatedPlanarCode(3, 3), my_error_model, RotatedPlanarRMPSDecoder(chi=8), 0.2))
