import qimpy as qp
import numpy as np


class System:
    '''TODO: document class System'''

    def __init__(self, *, rc, lattice, ions=None, symmetries=None,
                 electrons=None):
        '''
        Parameters
        ----------
        TODO
        '''
        self.rc = rc

        # Initialize lattice:
        self.lattice = qp.construct(qp.lattice.Lattice, lattice, 'lattice',
                                    rc=rc)

        # Initialize ions:
        if ions is None:
            # Set-up default of no ions:
            ions = {
                'pseudopotentials': [],
                'coordinates': []}
        self.ions = qp.construct(qp.ions.Ions, ions, 'ions',
                                 rc=rc)

        # Initialize symmetries:
        if symmetries is None:
            symmetries = {}
        self.symmetries = qp.construct(
            qp.symmetries.Symmetries, symmetries, 'symmetries',
            rc=rc, lattice=self.lattice, ions=self.ions)

        # Initialize electrons:
        self.electrons = qp.construct(
            qp.electrons.Electrons, electrons, 'electrons',
            rc=rc, lattice=self.lattice, symmetries=self.symmetries)


def construct(Class, params, object_name,
              **kwargs):
    '''Construct an object of type Class from params and kwargs
    if params is a dict, and just from kwargs if params is None.
    Any hyphens in keys within params are replaced with _ for convenience.
    Otherwise check that params is already of type Class, and if not,
    raise an error clearly stating what all types object_name can be.'''

    # Try all the valid possibilities:
    if isinstance(params, dict):
        return Class(
            **kwargs,
            **dict((k.replace('-', '_'), v) for k, v in params.items()))
    if params is None:
        return Class(**kwargs)
    if isinstance(params, Class):
        return params

    # Report error with canonicalized class name:
    module = Class.__module__
    module_elems = ([] if module is None else (
        [elem for elem in module.split('.')
         if not elem.startswith('_')]))  # drop internal module names
    module_elems.append(Class.__qualname__)
    class_name = '.'.join(module_elems)
    raise TypeError(object_name + ' must be dict or ' + class_name)
