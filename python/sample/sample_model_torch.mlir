module {
  func.func @main(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,5],f32> {
    %int0 = torch.constant.int 0
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>) : !torch.vtensor<[5],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_5_4_torch.float32> : tensor<5x4xf32>) : !torch.vtensor<[5,4],f32>
    %2 = torch.vtensor.literal(dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>) : !torch.vtensor<[3,4],f32>
    %int1 = torch.constant.int 1
    %3 = torch.aten.add.Tensor %arg0, %2, %int1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    %4 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[5,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,5],f32>
    %5 = torch.aten.mm %3, %4 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,5],f32> -> !torch.vtensor<[3,5],f32>
    %6 = torch.aten.add.Tensor %5, %0, %int1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5],f32>, !torch.int -> !torch.vtensor<[3,5],f32>
    %7 = torch.aten.clamp %6, %float0.000000e00, %float1.000000e00 : !torch.vtensor<[3,5],f32>, !torch.float, !torch.float -> !torch.vtensor<[3,5],f32>
    return %7 : !torch.vtensor<[3,5],f32>
  }
}