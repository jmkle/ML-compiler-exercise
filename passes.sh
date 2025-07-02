###  Pipeline to get from linalg to llvm dialect  ###
mlir-opt --canonicalize --convert-elementwise-to-linalg --convert-tensor-to-linalg --one-shot-bufferize=bufferize-function-boundaries --buffer-deallocation-pipeline --convert-linalg-to-loops --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts --convert-func-to-llvm --canonicalize --sccp --cse --symbol-dce $PWD/python/sample_model_linalg.mlir
# or
build-ninja/tools/tutorial-opt --linalg-to-llvm $PWD/python/sample_model_linalg.mlir > $PWD/python/sample_model_llvm.mlir


###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/python/sample_model_llvm.mlir > $PWD/python/sample_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/python/sample_model_llvm_ir.ll

###  Compile  ###
gcc -c python/sample_model_main.cpp -o python/sample_model_main.o && gcc python/sample_model_main.o python/sample_model_llvm_ir.o -o python/a.out