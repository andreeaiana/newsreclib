Installation
============

NewsRecLib requires Python version 3.9 or later.

NewsRecLib requires PyTorch, PyTorch Lightning, and TorchMetrics version 2.0 or later.
If you want to use NewsRecLib with GPU, please ensure CUDA or cudatoolkit version of 11.8.

Install from source
-------------------
CONDA
^^^^^

.. code:: bash

   git clone git@github.com:andreeaiana/newsreclib.git
   cd newsreclib
   conda create --name newsreclib_env python=3.8
   conda activate
   pip install -e .
