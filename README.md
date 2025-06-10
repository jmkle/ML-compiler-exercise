Initalize the submodules (torch-mlir, llvm-project)
`git submodule update --init --recursive`

Set-up a python virtual environment
```
python3 -m venv venv_torch_mlir
source venv_torch_mlir/bin/activate
pip install --upgrade pip
```

Install latest requirements
`python -m pip install -r requirements.txt -r torchvision-requirements.txt`

Configuration for building (in torch-mlir)
```
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
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD"
```

Build (and inital testing)
ToDo: Use Ninja directly
`cmake --build build --target check-torch-mlir --target check-mlir --target check-torch_mlir-python`