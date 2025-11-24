# Chapter 1: Introduction and Project Setup

Credits: The blog article by [Jeremy Kun](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) helped me a lot to build this pipeline, especially [this article](https://www.jeremykun.com/2023/11/01/mlir-lowering-through-llvm/) on lowering MLIR dialects.

For a general introduction, I highly recommend Stephan Diehl's [MLIR GPU lowering posts](https://www.stephendiehl.com/tags/mlir/).

## Introduction

MLIR is increasingly used in the compilation process of machine learning models. Unlike other compilers, IREE's compilation process has multiple phases in which it can represent its IR in different stages. These stages are different representations of the model with their own set of instructions and are referred to as _MLIR dialects_. A classification of dialects can be found [here](https://youtu.be/hIt6J1_E21c?t=800). This multi-phase approach allows for multiple levels of abstraction and, with that, better ways to target devices—something that is much harder to achieve in traditional two-phase compilers. Different optimizations can be applied at different abstraction levels where they are best suited.

Usually, you have a program that imports your models from their defining framework (e.g., PyTorch, TensorFlow, ONNX) to a high-level MLIR dialect, such as the linalg dialect. For example, torch-mlir reads the PyTorch model's graph representation and creates a representation in the torch dialect (which can also be linalg). Then the power of MLIR comes into play. Many passes exist to incrementally lower the dialects from high-level ones to more assembly-like dialects (so-called exit dialects, e.g., llvm). Along the way, many optimizations can also be performed on the corresponding dialects. One can also define custom dialects, passes, etc. Different types of passes exist: transform passes transform code within a dialect, conversion passes convert code between dialects, and analysis passes include things like data-flow analysis.

Once at an exit dialect, we need to translate to LLVM IR, thereby leaving the MLIR space. From here, object code can be generated for a specific target. Then the model can be called from, for example, C++ via a regular function call.

## A State-of-the-Art ML Compiler: IREE

A large and widely used state-of-the-art MLIR-based ML compiler is [IREE](https://iree.dev/). It consists of an end-to-end compiler and runtime. The generated IR is optimized for real-time machine learning inference on edge/mobile devices running on various hardware accelerators. The first step is to build the model in the framework of choice, e.g., TensorFlow or PyTorch. The source model from the frontend is then fed into the IREE compiler. The compiler builds different shared library-like modules called artifacts containing the executable binary for target backends (the execution logic) and the partitioned host program (the scheduling logic), among other components. The last component in the IREE workflow is the IREE runtime, which dynamically links and loads modules into isolated contexts. With a Hardware Abstraction Layer (HAL), it can easily make use of hardware backends and provides multiple APIs for ease of use.

The following describes [IREE's project architecture](https://iree.dev/#project-architecture) depicted below. First, the created model is translated into an MLIR representation by the framework. It takes a snapshot of the structure and data constituting the model and translates it into an MLIR dialect that IREE accepts as input. For example, the dialect for TensorFlow Lite is TOSA. `linalg` and `arith` depicted in the figure are also valid input dialects. These dialects are then fed into the compiler. (For example, the IREE importer for PyTorch models is [iree-turbine](https://github.com/iree-org/iree-turbine).)

The IREE compiler stack has 3 stages that help lower the operations, each being an Intermediate Representation (or MLIR dialect). The first is the flow dialect, which captures high-level model structure and divides computations into dispatchable regions that can later be optimized and scheduled. In the figure, the partitioned workload can be seen. The next stage enables efficient execution of machine learning models by modeling workloads as streams of data and operations that can be scheduled and executed asynchronously and in parallel. This makes workloads well-suited for high-performance machine learning and other compute-heavy applications and is crucial for optimizing the execution of large, complex workloads where managing computation and data flow efficiently is essential.

The Hardware Abstraction Layer (HAL) is the third layer in the IREE compiler, managing buffers and defining how operations should be executed on different devices. IREE supports multiple target backends for different hardware architectures. These backends lower the intermediate representation into code that is optimized for a specific hardware target. For example, the LLVM backend targets CPUs and is optimized for scalar and vectorized execution on a variety of architectures like x86 and ARM. For targeting GPUs (e.g., Vulkan, OpenCL), the SPIR-V backend is used, which lowers the IR into SPIR-V, a widely used intermediate representation for GPU execution.

As mentioned in the overview, the compiler produces IREE modules that are then forwarded to the lightweight IREE runtime. The runtime efficiently executes compiled models on different hardware and consists of several plugins, a HAL, and an IREE Virtual Machine (VM). Furthermore, API bindings are provided for IREE's compiler and runtime to be used in various programming languages. The number of supported languages is under development and therefore still growing. To run individual compiler passes, translations, and other transformations step by step, IREE provides tools for these tasks. Some examples are shown in the figure.

![The IREE project architecture](https://iree.dev/assets/images/iree_architecture_dark.svg#gh-dark-mode-only)

## What This Tutorial Is About

In this tutorial, we will build our own MLIR pipeline to lower real ML models. Specifically, we take models from PyTorch, import them with [torch-mlir](https://github.com/llvm/torch-mlir), and then select multiple passes to lower the model to the MLIR LLVM dialect. We'll then export the model to x86 assembly code to be called by C/C++. We will also write our own pass that converts matrix multiplications to OpenBLAS function calls to speed up execution.

## The Project Setup

The project setup can be found [here](link). It contains the following directories:

- **externals** includes torch-mlir and, with that, LLVM/MLIR in-tree
- **lib** to store our own passes
- **python** that imports the PyTorch models to MLIR
- **tools** contains our pass pipeline

Aside: Usually, you also have an `include/` folder that holds header files whereas `lib/` holds implementation files. `lib/` also has subdirectories for different kinds of passes we mentioned above (e.g., `lib/Transform`). But we put headers and implementations together in `lib/` as we don't have many passes (currently only 1).

## Getting Started

First, we need to initialize the submodules (torch-mlir, llvm-project): `git submodule update --init --recursive`

Set up a Python virtual environment to install the latest requirements from torch-mlir:

```bash
python3 -m venv .venv_torch_mlir
source .venv_torch_mlir/bin/activate
pip install --upgrade pip

# Go to torch-mlir directory and install requirements
cd externals/torch-mlir
python -m pip install -r requirements.txt -r torchvision-requirements.txt
```

We can then configure the build for torch-mlir (we build llvm/mlir in-tree):

```bash
cmake -GNinja -Bbuild \
  `# Enables "--debug" and "--debug-only" flags for the "torch-mlir-opt" tool` \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  `# For building LLVM "in-tree"` \
  externals/llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON
```

Build (and initial testing): `cmake --build build --target check-torch-mlir`
Or use Ninja directly: `ninja -C build check-torch-mlir`

In `.venv_torch_mlir/bin/activate`, add the following (adapt the path if necessary):

```bash
# Add torch-mlir-opt to PATH
export PATH="/home/ab123456/ml-compiler-exercise/externals/torch-mlir/build/bin/:$PATH"

# Add MLIR Python bindings and setup Python environment to export the built Python packages
export PYTHONPATH=/home/ab123456/ml-compiler-exercise/externals/torch-mlir/build/tools/mlir/python_packages/mlir_core:/home/ab123456/ml-compiler-exercise/externals/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:/home/ab123456/ml-compiler-exercise/externals/torch-mlir/test/python/fx_importer
```

In the next chapter, we will write simple example PyTorch models and import them to torch-mlir to view the model in the MLIR torch dialect.
