# GPU compilation pipeline (in progress...)

Based on [Stephan Diehl's GPU pipeline](https://github.com/sdiehl/gpu-offload/tree/main)

## Requirements:

`pip install mlir_python_bindings -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest`

`pip install cuda-python==12.6.0` (install cuda-python 12.x.x, if CUDA 12.x is installed, otherwise `pip install cuda-python` is sufficient)