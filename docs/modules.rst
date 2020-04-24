Main module (pydmd.dmd)
=======================

The dynamic mode decomposition is a regression technique for discovering the global dynamical operator describing
a state space dynamical system from measurement data. The discovered operator is decomposed into the dominant
spatiotemporal coherent structures stored as eigenvalues and DMD modes.

.. automodule:: pydmd.dmd

DMD
---
Dynamic mode decomposition.

.. autofunction:: pydmd.dmd.DMD.__init__

.. autoclass:: pydmd.dmd.DMD
    :members:

DMDc
----
Dynamic mode decomposition with control.

.. autofunction:: pydmd.dmd.DMDc.__init__

.. autoclass:: pydmd.dmd.DMDc
    :members:

BiDMD
-----
Bilinear dynamic mode decomposition.

.. autofunction:: pydmd.dmd.biDMD.__init__

.. autoclass:: pydmd.dmd.biDMD
    :members:

BiDMDc
------
Bilinear dynamic mode decomposition with control.

.. autofunction:: pydmd.dmd.biDMDc.__init__

.. autoclass:: pydmd.dmd.biDMDc
    :members:


Data processing utilities (pydmd.process)
=========================================

.. automodule:: pydmd.process
    :members:


Plotting utilities (pydmd.plot)
===============================

.. automodule:: pydmd.plot
    :members:

