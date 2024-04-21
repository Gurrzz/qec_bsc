import functools
import itertools
import json
import logging
import operator

import numpy as np
from mpmath import mp

from qecsim import paulitools as pt, tensortools as tt
from qecsim.model import Decoder, cli_description
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarRMPSDecoder

logger = logging.getLogger(__name__)


@cli_description('Rotated MPS ([chi] INT >=0, [mode] CHAR, ...)')
class RotatedPlanarG81RMPSDecoder(RotatedPlanarRMPSDecoder):
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
        def h_node_value(self, prob_dist, f, n, e, s, w, even_column):
            """Return horizontal edge tensor element value."""
            paulis = ('I', 'X', 'Y', 'Z')
            op_to_pr = dict(zip(paulis, prob_dist))
            f = pt.pauli_to_bsf(f)
            I, X, Y, Z = pt.pauli_to_bsf(paulis)

            # n, e, s, w are in {0, 1} so multiply op to turn on or off
            if (even_column):
                op = (f + (n * Z) + (e * X) + (s * Z) + (w * X)) % 2
            else:
                op = (f + (n * X) + (e * Z) + (s * X) + (w * Z)) % 2

            return op_to_pr[pt.bsf_to_pauli(op)]


        @functools.lru_cache()
        def v_node_value(self, prob_dist, f, n, e, s, w, even_column):
            """Return vertical edge tensor element value."""
            # N.B. for v_node order of nesw is rotated relative to h_node
            return self.h_node_value(prob_dist, f, e, s, w, n, even_column)


        @functools.lru_cache(maxsize=256)
        def create_q_node(self, prob_dist, f, h_node, even_column, compass_direction=None):
            """Create q-node for tensor network.

            Notes:

            * H-nodes have Z-plaquettes above and below (i.e. in NE and SW directions).
            * V-nodes have Z-plaquettes on either side (i.e. in NW and SE directions).
            * Columns are considered even/odd according to indexing defined in :class:`RotatedPlanarCode`.

            :param h_node: If H-node, else V-node.
            :type h_node: bool
            :param prob_dist: Probability distribution in the format (Pr(I), Pr(X), Pr(Y), Pr(Z)).
            :type prob_dist: (float, float, float, float)
            :param f: Pauli operator on qubit as 'I', 'X', 'Y', or 'Z'.
            :type f: str
            :param even_column: If even column, else odd column.
            :type even_column: bool
            :param compass_direction: Compass direction as 'n', 'ne', 'e', ..., 'nw', or falsy for bulk.
            :type compass_direction: str
            :return: Q-node for tensor network.
            :rtype: numpy.array (4d)
            """

            # H indicates h-node with shape (n,e,s,w).
            # * indicates delta nodes with shapes (n,I,j), (e,J,k), (s,K,l), (w,L,i) for n-, e-, s-, and w-deltas
            #   respectively.
            # n,e,s,w,i,j,k,I,J,K are bond labels
            #
            #   i     I
            #   |     |
            # L-*     *-j
            #    \   /
            #    w\ /n
            #      H
            #    s/ \e
            #    /   \
            # l-*     *-J
            #   |     |
            #   K     k
            #
            # Deltas are absorbed into h-node over n,e,s,w legs and reshaped as follows:
            # nesw -> (iI)(jJ)(Kk)(Ll)

            # define shapes # q_node:(n, e, s, w); delta_nodes: n:(n,I,j), e:(e,J,k), s:(s,K,l), w:(w,L,i)
            if h_node:
                # bulk h-node
                q_shape = (2, 2, 2, 2)
                if even_column:
                    n_shape, e_shape, s_shape, w_shape = (2, 2, 2), (2, 1, 2), (2, 2, 2), (2, 1, 2)
                else:
                    n_shape, e_shape, s_shape, w_shape = (2, 2, 1), (2, 2, 2), (2, 2, 1), (2, 2, 2)
                # modifications for directions
                if compass_direction == 'n':
                    q_shape = (2, 2, 2, 1)
                    n_shape, w_shape = (2, 1, 2), (1, 1, 1)
                elif compass_direction == 'ne':
                    q_shape = (1, 2, 2, 1)
                    n_shape, e_shape, w_shape = (1, 1, 1), (2, 1, 2), (1, 1, 1)
                elif compass_direction == 'e':
                    q_shape = (1, 2, 2, 2)
                    n_shape, e_shape = (1, 1, 1), (2, 1, 2)
                elif compass_direction == 'se':  # always even
                    q_shape = (1, 1, 2, 2)
                    n_shape, e_shape, s_shape = (1, 1, 1), (1, 1, 1), (2, 1, 2)
                elif compass_direction == 's':  # always even
                    q_shape = (2, 1, 2, 2)
                    e_shape, s_shape = (1, 1, 1), (2, 1, 2)
                elif compass_direction == 'sw':  # always even
                    q_shape = (2, 1, 1, 2)
                    e_shape, s_shape, w_shape = (1, 1, 1), (1, 1, 1), (2, 1, 2)
                elif compass_direction == 'w':  # always even
                    q_shape = (2, 2, 1, 2)
                    s_shape, w_shape = (1, 1, 1), (2, 1, 2)
                elif compass_direction == 'nw':  # always even
                    q_shape = (2, 2, 1, 1)
                    n_shape, s_shape, w_shape = (2, 1, 2), (1, 1, 1), (1, 1, 1)
            else:
                # bulk v-node
                q_shape = (2, 2, 2, 2)
                if even_column:
                    n_shape, e_shape, s_shape, w_shape = (2, 2, 2), (2, 1, 2), (2, 2, 2), (2, 1, 2)
                else:
                    n_shape, e_shape, s_shape, w_shape = (2, 2, 1), (2, 2, 2), (2, 2, 1), (2, 2, 2)
                # modifications for directions
                if compass_direction == 'n':
                    q_shape = (1, 2, 2, 2)
                    n_shape, w_shape = (1, 1, 1), (2, 2, 1)
                elif compass_direction == 'ne':
                    q_shape = (1, 1, 2, 2)
                    n_shape, e_shape, w_shape = (1, 1, 1), (1, 1, 1), (2, 2, 1)
                elif compass_direction == 'e':
                    q_shape = (2, 1, 2, 2)
                    n_shape, e_shape = (2, 2, 1), (1, 1, 1)
                elif compass_direction == 'se':  # always odd
                    q_shape = (2, 1, 1, 2)
                    n_shape, e_shape, s_shape = (2, 2, 1), (1, 1, 1), (1, 1, 1)
                elif compass_direction == 's':  # always odd
                    q_shape = (2, 2, 1, 2)
                    e_shape, s_shape = (2, 2, 1), (1, 1, 1)
                elif compass_direction == 'sw':  # not possible
                    raise ValueError('Cannot have v-node in SW corner of lattice.')
                elif compass_direction == 'w':  # always even
                    q_shape = (2, 2, 2, 1)
                    s_shape, w_shape = (2, 2, 1), (1, 1, 1)
                elif compass_direction == 'nw':  # always even
                    q_shape = (1, 2, 2, 1)
                    n_shape, s_shape, w_shape = (1, 1, 1), (2, 2, 1), (1, 1, 1)

            # create deltas
            n_delta = tt.tsr.delta(n_shape)
            e_delta = tt.tsr.delta(e_shape)
            s_delta = tt.tsr.delta(s_shape)
            w_delta = tt.tsr.delta(w_shape)
            # create q_node and fill values
            q_node = np.empty(q_shape, dtype=np.float64)
            for n, e, s, w in np.ndindex(q_node.shape):
                if h_node:
                    q_node[(n, e, s, w)] = self.h_node_value(prob_dist, f, n, e, s, w, even_column)
                else:
                    q_node[(n, e, s, w)] = self.v_node_value(prob_dist, f, n, e, s, w, even_column)
            # derive combined node shape
            shape = (w_shape[2] * n_shape[1], n_shape[2] * e_shape[1], e_shape[2] * s_shape[1], s_shape[2] * w_shape[1])
            # create combined node by absorbing deltas into q_node: nesw -> (iI)(jJ)(Kk)(Ll)
            node = np.einsum('nesw,nIj,eJk,sKl,wLi->iIjJKkLl', q_node, n_delta, e_delta, s_delta, w_delta).reshape(
                shape)
            # return combined node
            return node

