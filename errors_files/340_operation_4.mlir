
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 2)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<256x2x4x512xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<256x7x15x512xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<256x7x15x512xf32>) -> tensor<256x7x15x512xf32>
    %2 = bufferization.alloc_tensor() : tensor<3x3xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<3x3xf32>) -> tensor<3x3xf32>
    %4 = bufferization.alloc_tensor() : tensor<256x3x7x512xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<256x3x7x512xf32>) -> tensor<256x3x7x512xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_3"} ins(%1, %3 : tensor<256x7x15x512xf32>, tensor<3x3xf32>) outs(%5 : tensor<256x3x7x512xf32>) -> tensor<256x3x7x512xf32>
    %8 = bufferization.alloc_tensor() : tensor<256x3x7x512xf32>
    %9 = bufferization.alloc_tensor() : tensor<256x3x7x512xf32>
    %10 = bufferization.alloc_tensor() : tensor<1x1xf32>
    %11 = bufferization.alloc_tensor() : tensor<256x2x4x512xf32>
    %12 = scf.forall (%arg0, %arg1, %arg2) in (16, 2, 256) shared_outs(%arg3 = %11) -> (tensor<256x2x4x512xf32>) {
      %15 = affine.apply #map(%arg0)
      %16 = affine.apply #map1(%arg2)
      %17 = affine.apply #map(%arg0)
      %18 = affine.apply #map1(%arg1)
      %19 = affine.apply #map1(%arg2)
      %20 = affine.apply #map(%arg0)
      %21 = affine.apply #map1(%arg1)
      %22 = affine.apply #map1(%arg2)
      %23 = affine.apply #map(%arg0)
      %24 = affine.apply #map1(%arg1)
      %25 = affine.apply #map1(%arg2)
      %extracted_slice = tensor.extract_slice %7[%17, %18, 0, %19] [16, 1, 7, 2] [1, 1, 1, 1] : tensor<256x3x7x512xf32> to tensor<16x1x7x2xf32>
      %extracted_slice_0 = tensor.extract_slice %8[%20, %21, 0, %22] [16, 1, 7, 2] [1, 1, 1, 1] : tensor<256x3x7x512xf32> to tensor<16x1x7x2xf32>
      %extracted_slice_1 = tensor.extract_slice %9[%23, %24, 0, %25] [16, 1, 7, 2] [1, 1, 1, 1] : tensor<256x3x7x512xf32> to tensor<16x1x7x2xf32>
      %26 = linalg.add {tag = "operation_4"} ins(%extracted_slice, %extracted_slice_0 : tensor<16x1x7x2xf32>, tensor<16x1x7x2xf32>) outs(%extracted_slice_1 : tensor<16x1x7x2xf32>) -> tensor<16x1x7x2xf32>
      %extracted_slice_2 = tensor.extract_slice %arg3[%15, %arg1, 0, %16] [16, 1, 4, 2] [1, 1, 1, 1] : tensor<256x2x4x512xf32> to tensor<16x1x4x2xf32>
      %27 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_5"} ins(%26, %10 : tensor<16x1x7x2xf32>, tensor<1x1xf32>) outs(%extracted_slice_2 : tensor<16x1x4x2xf32>) -> tensor<16x1x4x2xf32>
      %28 = affine.apply #map(%arg0)
      %29 = affine.apply #map1(%arg2)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %27 into %arg3[%28, %arg1, 0, %29] [16, 1, 4, 2] [1, 1, 1, 1] : tensor<16x1x4x2xf32> into tensor<256x2x4x512xf32>
      }
    }
    %13 = call @nanoTime() : () -> i64
    %14 = arith.subi %13, %6 : i64
    call @printI64(%14) : (i64) -> ()
    call @printNewline() : () -> ()
    return %12 : tensor<256x2x4x512xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<256x2x4x512xf32>
    }
    return
  }
}
