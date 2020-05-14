Main module (dmdlab.dmd)
========================

The dynamic mode decomposition is a regression technique for discovering the global dynamical operator describing
a state space dynamical system from measurement data. The discovered operator is decomposed into the dominant
spatiotemporal coherent structures stored as eigenvalues and DMD modes.

.. automodule:: dmdlab.dmd

DMD
---
Dynamic mode decomposition.

.. autofunction:: dmdlab.dmd.DMD.__init__

.. autoclass:: dmdlab.dmd.DMD
    :members:

DMDc
----
Dynamic mode decomposition with control.

.. autofunction:: dmdlab.dmd.DMDc.__init__

.. autoclass:: dmdlab.dmd.DMDc
    :members:

BiDMD
-----
Bilinear dynamic mode decomposition.

.. autofunction:: dmdlab.dmd.biDMD.__init__

.. autoclass:: dmdlab.dmd.biDMD
    :members:

BiDMDc
------
Bilinear dynamic mode decomposition with control.

.. autofunction:: dmdlab.dmd.biDMDc.__init__

.. autoclass:: dmdlab.dmd.biDMDc
    :members:


Data processing utilities (dmdlab.process)
==========================================

.. automodule:: dmdlab.process
    :members:


Plotting utilities (dmdlab.plot)
================================

.. automodule:: dmdlab.plot
    :members:

