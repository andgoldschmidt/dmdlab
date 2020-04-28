dmdlab
======
Dynamic Mode Decomposition in Python
------------------------------------

Dynamic Mode Decomposition (DMD) is an algorithm for determining an equation-free representation of a dynamical system
based on dominant spatiotemporal coherent structures.

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


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    notebooks/Examples.ipynb
    modules
    license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
