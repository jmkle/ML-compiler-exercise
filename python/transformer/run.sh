../../externals/torch-mlir/build/bin/torch-mlir-opt \
  -torch-backend-to-linalg-on-tensors-backend-pipeline \
  $PWD/rn18.mlir > $PWD/rn18_linalg.mlir

  #-canonicalize \
  #-torch-decompose-complex-ops \
  #-torch-simplify-shape-calculations \
  #-torch-simplify-dtype-calculations \
  #-convert-torch-to-linalg \
  #-convert-torch-to-arith \
  #-canonicalize \
  #-torch-backend-to-linalg-on-tensors-backend-pipeline \