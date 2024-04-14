import operator

from qecsim.model import cli_description
from qecsim.models.rotatedplanar import RotatedPlanarCode
import RotatedPlanarXZPauli


@cli_description('Rotated planar G81 (distance INT odd >= 3)')
class RotatedPlanarXZCode(RotatedPlanarCode):
    r"""
    Implements a rotated planar mixed boundary code with XZXZ/ZXZX plaquettes defined by its lattice size.

    In addition to the members defined in :class:`qecsim.model.StabilizerCode`, it provides several lattice methods as
    described below.

    Lattice methods:

    * Get size: :meth:`size`.
    * Get plaquette type: :meth:`is_virtual_plaquette`.
    * Get and test bounds: :meth:`site_bounds`, :meth:`is_in_site_bounds`, :meth:`is_in_plaquette_bounds`.
    * Resolve a syndrome to plaquettes: :meth:`syndrome_to_plaquette_indices`.
    * Construct a Pauli operator on the lattice: :meth:`new_pauli`.

    Indices:

    * Indices are in the format (x, y).
    * Qubit sites (vertices) are indexed by (x, y) coordinates with the origin at the lower left qubit.
    * Stabilizer plaquettes are indexed by (x, y) coordinates such that the lower left corner of the plaquette is on the
      qubit site at (x, y). 
    * XZ/XZ-type stabilizer plaquette indices satisfy (x-y) % 2 == 1.
    * ZX/ZX-type stabilizer plaquette indices satisfy (x-y) % 2 == 0. 

    For example, qubit site indices on a 3 x 3 lattice:
    ::

             (0,2)-----(1,2)-----(2,2)
               |         |         |
               |         |         |
               |         |         |
             (0,1)-----(1,1)-----(2,1)
               |         |         |
               |         |         |
               |         |         |
             (0,0)-----(1,0)-----(2,0)

    For example, stabilizer plaquette indices on a 3 x 3 lattice:
    ::

                 -------
                /       \
               |Z (0,2) X|
               +---------+---------+-----
               |X       Z|X       Z|X    \
               |  (0,1)  |  (1,1)  |(2,1) |
               |X       Z|X       Z|X    /
          -----+---------+---------+-----
         /    X|Z       X|Z       X|
        |(-1,0)|  (0,0)  |  (1,0)  |
         \    X|Z       X|Z       X|
          -----+---------+---------+
                         |X       Z|
                          \ (1,-1)/
                           -------
    """

    def __init__(self, distance):
        """
        Initialise new rotated planar G81 (XZXZ/ZXZX) code.

        :param distance: Number of rows/columns in lattice.
        :type distance: int
        :raises ValueError: if size smaller than 3.
        :raises ValueError: if size is even.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if operator.index(distance) < self.MIN_SIZE[0]:
                raise ValueError('{} minimum distance is {}.'.format(type(self).__name__, self.MIN_SIZE[0]))
            if distance % 2 == 0:
                raise ValueError('{} size must be odd.'.format(type(self).__name__))
        except TypeError as ex:
            raise TypeError('{} invalid parameter type'.format(type(self).__name__)) from ex
        super().__init__(distance, distance)

    # < StabilizerCode interface methods >

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return 'Rotated planar XZ {}'.format(self.n_k_d[2])

    # </ StabilizerCode interface methods >

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.n_k_d[2])

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of planar Pauli for this code.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        :return: Rotated planar XZ Pauli
        :rtype: RotatedPlanarXZPauli
        """
        return RotatedPlanarXZPauli(self, bsf)

