import functools
import logging

from qecsim.model import cli_description
from qecsim.models.rotatedplanar import RotatedPlanarRMPSDecoder

logger = logging.getLogger(__name__)


@cli_description('Rotated MPS ([chi] INT >=0, [mode] CHAR, ...)')
class RotatedPlanarG18RMPSDecoder(RotatedPlanarRMPSDecoder):
    r"""
    Implements a rotated planar G18 (XZXZ/ZXZX) Rotated Matrix Product State (RMPS) decoder.

    Decoding algorithm:

    * A sample recovery operation :math:`f` is found by applying a path of X or Z operators between each plaquette,
      identified by the syndrome, along a diagonal to an appropriate boundary.
    * The probability of the left coset :math:`fG` of the stabilizer group :math:`G` of the planar code with respect
      to :math:`f` is found by contracting an appropriately defined MPS-based tensor network (see
      https://arxiv.org/abs/1405.4883).
    * Since this is a rotated MPS decoder, the links of the network are rotated 45 degrees by splitting each stabilizer
      node into 4 delta nodes that are absorbed into the neighbouring qubit nodes.
    * The complexity of the algorithm can managed by defining a bond dimension :math:`\chi` to which the MPS bond
      dimension is truncated after each row/column of the tensor network is contracted into the MPS.
    * The probability of cosets :math:`f\bar{X}G`, :math:`f\bar{Y}G` and :math:`f\bar{Z}G` are calculated similarly.
    * The default contraction is column-by-column but can be set using the mode parameter to row-by-row or the average
      of both contractions.
    * A sample recovery operation from the most probable coset is returned.

    Notes:

    * Specifying chi=None gives an exact contract (up to rounding errors) but is exponentially slow in the size of
      the lattice.
    * Modes:

        * mode='c': contract by columns
        * mode='r': contract by rows
        * mode='a': contract by columns and by rows and, for each coset, take the average of the probabilities.

    * Contracting by columns (i.e. truncating vertical links) may give different coset probabilities to contracting by
      rows (i.e. truncating horizontal links). However, the effect is symmetric in that transposing the sample_pauli on
      the lattice and exchanging X and Z single Paulis reverses the difference between X and Z cosets probabilities.

    Tensor network example:

    3x3 rotated planar code with H or V indicating qubits and plaquettes indicating stabilizers:
    ::

           /---\
           |   |
           H---V---H--\
           |   |   |  |
           |   |   |  |
           |   |   |  |
        /--V---H---V--/
        |  |   |   |
        |  |   |   |
        |  |   |   |
        \--H---V---H
               |   |
               \---/


    MPS tensor network as per https://arxiv.org/abs/1405.4883 (s=stabilizer), except H and V qubit tensors are defined
    identically with the NE and SW (NW and SE) stabilizers applying Z (X) operators: (BEHÖVER ÄNDRAS)
    ::

             s
            / \
           H   V   H
            \ / \ / \
             s   s   s
            / \ / \ /
           V   H   V
          / \ / \ /
         s   s   s
          \ / \ / \
           H   V   H
                \ /
                 s

    Links are rotated by splitting stabilizers and absorbing them into neighbouring qubits.
    For even columns of stabilizers (according to indexing defined in
    :class:`qecsim.models.planar.RotatedPlanarXZCode`), a 'lucky' horseshoe shape is used:
    ::

        H   V      H     V
         \ /        \   /       H V
          s    =>    s s    =>  | |
         / \         | |        V-H
        V   H        s-s
                    /   \
                   V     H

    For odd columns, an 'unlucky' horseshoe shape is used:
    ::

        H   V      H     V
         \ /        \   /       H-V
          s    =>    s-s    =>  | |
         / \         | |        V H
        V   H        s s
                    /   \
                   V     H

    Resultant MPS tensor network, where horizontal (vertical) bonds have dimension 2 (4) respectively.
    ::

          0 1 2
        0 H-V-H
          | | |
        1 V-H-V
          | | |
        2 H-V-H
    """

    @classmethod
    def sample_recovery(cls, code, syndrome):
        """
        Return a sample Pauli consistent with the syndrome, created by applying a path of X or Z operators between each
        plaquette, identified by the syndrome.

        :param code: Rotated planar G18 (XZXZ/ZXZX) code.
        :type code: RotatedPlanarG18Code
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Sample recovery operation as rotated planar pauli.
        :rtype: RotatedPlanarG18Pauli
        """
        # prepare a new blank sample
        sample_recovery = code.new_pauli()
        # ask code for plaquette_indices associated with the non-commuting stabilizers identified by the syndrome
        plaquette_indices = code.syndrome_to_plaquette_indices(syndrome)

        max_site_x, max_site_y = code.site_bounds
        for plaq_index in plaquette_indices:
            # NOTE: plaquette index coincides with the index of the site in its lower left corner
            plaq_x, plaq_y = plaq_index

            # If the diagonal is even counting out from the center diagonal from bottom left to top right
            if ((plaq_x - plaq_y) % 2 == 0):
                
                # Even row => ZX/ZX plaquette
                if (plaq_y % 2 == 0): 

                    # Add an (X)ZXZ... path from the site at the bottom left corner of the plaquette and left to the boundary
                    for x in reversed(range(0, plaq_x + 1)):
                        if (x % 2 == 0):
                            sample_recovery.site('X', (x, plaq_y))
                        else:
                            sample_recovery.site('Z', (x, plaq_y))

                # Odd row => XZ/XZ plaquette
                else:

                    # Add a ZXZX... path from the site at the top left corner of the plaquette and left to the boundary
                    for x in reversed(range(0, plaq_x + 1)):
                        if (x % 2 == 0):
                            sample_recovery.site('X', (x, plaq_y + 1))
                        else:
                            sample_recovery.site('Z', (x, plaq_y + 1))

            # If the diagonal is odd counting out from the center diagonal from bottom left to top right
            else:

                # Add a ZZZ... path from the lower left, down to the boundary
                if (plaq_x % 2 == 0):
                    for y in range(0, plaq_y + 1):
                        sample_recovery.site('Z', (plaq_x, y))

                # Add a ZZZ... path from the lower right, down to the boundary
                else:
                    for y in range(0, plaq_y + 1):
                        sample_recovery.site('Z', (plaq_x + 1, y))

        # return sample
        return sample_recovery


    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        params = [('chi', self._chi), ('mode', self._mode), ('tol', self._tol), ]
        return 'Rotated planar G18 (XZXZ/ZXZX) RMPS ({})'.format(', '.join('{}={}'.format(k, v) for k, v in params if v))

    class TNC(RotatedPlanarRMPSDecoder.TNC):
        """Tensor network creator"""

        @functools.lru_cache()
        def v_node_value(self, prob_dist, f, n, e, s, w):
            """Return V-node qubit tensor element value."""
            # N.B. with XZ/ZX plaquettes, H-node and V-node values are both as per H-node values of the CSS code
            return self.h_node_value(prob_dist, f, n, e, s, w)

