from qecsim.models.rotatedplanar import RotatedPlanarPauli

class RotatedPlanarG81Pauli(RotatedPlanarPauli): 
    """
    Defines a Pauli operator on a rotated planar Clifford deformed G81 lattice. 

    Notes:

    * This is a utility class used by rotated planar implementations of the core models.
    * It is typically instantiated using :meth:`qecsim.models.rotatedplanar.RotatedPlanarCode.new_pauli`//BEHÖVER ÄNDRAS

    Use cases:

    * Construct a rotated planar Pauli operator by applying site, plaquette, and logical operators: :meth:`site`,
      :meth:`plaquette`, :meth:`logical_x`, :meth:`logical_z`.
    * Get the single Pauli operator applied to a given site: :meth:`operator`
    * Convert to binary symplectic form: :meth:`to_bsf`.
    * Copy a rotated planar Pauli G81 operator: :meth:`copy`.
    """
    def plaquette(self, index): 
        """
        Apply a plaquette operator at the given index.

        Notes:

        * Index is in the format (x, y).
        * If an ZX/ZX-type plaquette is indexed (i.e. (x - y) % 2 == 0), then Z operators are applied in the NW and SW 
          corners of the plaquette, and X operators are applied in the NE and SE corners of the plaquette.
        * If an XZ/XZ-type plaquette is indexed (i.e. (x - y) % 2 == 1), then X operators are applied in the NW and SW 
          corners of the plaquette, and Z operators are applied in the NE and SE corners of the plaquette.
        * Applying plaquette operators on plaquettes that lie outside the lattice have no effect on the lattice.

        :param index: Index identifying the plaquette in the location format (x, y).
        :type index: 2-tuple of int
        :return: self (to allow chaining)
        :rtype: RotatedPlanarPauli
        """
        x, y = index
        # apply if index within lattice
        if self.code.is_in_plaquette_bounds(index):

            # if ZX/ZX
            if (abs(x - y) % 2 == 0): 
                # flip plaquette sites
                self.site('Z', (x, y)) # SW
                self.site('Z', (x, y + 1)) # NW
                self.site('X', (x + 1, y + 1)) # NE
                self.site('X', (x + 1, y)) # SE

            # if XZ/XZ
            else:
                # flip plaquette sites
                self.site('X', (x, y)) # SW
                self.site('X', (x, y + 1)) # NW
                self.site('Z', (x + 1, y + 1)) # NE
                self.site('Z', (x + 1, y)) # SE

        return self


    def logical_x(self): 
        """
        Apply a logical X operator, i.e. <<INSERT OPERATOR STRUCTURE>>

        Notes: 

        * Operators are applied to the bottom row to allow optimisation of the MPS decoder. 
        
        :return: self (to allow chaining)
        :rtype: RotatedPlanarG81Pauli
        """

        max_site_x, max_site_y = self.code.site_bounds
        for x in range(0, max_site_x + 1):
            if x % 2 == 0:
                self.site('X', (x, 0))
            else:
                self.site('Z', (x, 0))

        return self


    def logical_z(self): 
        """
        Apply a logical Z operator, i.e. <<INSERT OPERATOR STRUCTURE>>

        Notes: 

        * Operators are applied to the rightmost column to allow optimisation of the MPS decoder. (KRAV ELLER VALBART?)

        :return: self (to allow chaining)
        :rtype: RotatedPlanarG81Pauli
        """
        
        max_site_x, max_site_y, = self.code.site_bounds
        self.site('Z', *((max_site_x, y) for y in range(0, max_site_y + 1)))

        return self


    # String representation of the class
    def __repr__(self): 
        return '{}({!r}, {!r})'.format(type(self).__name__, self.code, self.to_bsf())

































