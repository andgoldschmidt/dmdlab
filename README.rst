.. image:: https://readthedocs.org/projects/dmdlab/badge/?version=latest
   :target: https://dmdlab.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
  
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://lbesson.mit-license.org/
   :alt: MIT License
 
Dynamic mode decomposition (DMD)is a tool for analyzing the dynamics of nonlinear systems.

This is an experimental DMD codebase for research purposes.

Alternatively, check out `PyDMD <https://mathlab.github.io/PyDMD/>`_, a professionally maintained open source DMD
codebase for Python.

Installation:

.. code-block:: python

    pip install dmdlab

Usage:

.. code-block:: python

    from dmdlab import DMD, plot_eigs
    import numpy as np
    from scipy.linalg import logm

    # Generate toy data
    ts = np.linspace(0,6,50)

    theta = 1/10
    A_dst = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    A_cts = logm(A_dst)/(ts[1]-ts[0])

    x0 = np.array([1,0])
    X = np.vstack([expm(A_cts*ti)@x0 for ti in ts]).T

    # Fit model
    model = DMD.from_full(X, ts)

    # Print the eigenvalue phases
    print(np.angle(model.eigs))

    >>> [0.1, -0.1]


For a technical reference, check out the `DMD book <http://www.dmdbook.com/>`_.
