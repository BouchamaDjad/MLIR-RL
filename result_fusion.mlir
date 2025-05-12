#map = affine_map<(d0) -> (d0 * 2)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<256x120xf32> {
      %c2 = arith.constant 2 : index
      %c3072 = arith.constant 3072 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<1024x3072xf32>
      %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<1024x3072xf32>) -> tensor<1024x3072xf32>
      %2 = bufferization.alloc_tensor() : tensor<1024x28xf32>
      %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<1024x28xf32>) -> tensor<1024x28xf32>
      %4 = bufferization.alloc_tensor() : tensor<3072x28xf32>
      %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<3072x28xf32>) -> tensor<3072x28xf32>
      %6 = call @nanoTime() : () -> i64
      %7 = linalg.matmul_transpose_a {tag = "operation_3"} ins(%1, %3 : tensor<1024x3072xf32>, tensor<1024x28xf32>) outs(%5 : tensor<3072x28xf32>) -> tensor<3072x28xf32>
      %8 = bufferization.alloc_tensor() : tensor<28x256xf32>
      %9 = bufferization.alloc_tensor() : tensor<3072x256xf32>
      %10 = linalg.matmul {tag = "operation_4"} ins(%7, %8 : tensor<3072x28xf32>, tensor<28x256xf32>) outs(%9 : tensor<3072x256xf32>) -> tensor<3072x256xf32>
      %11 = bufferization.alloc_tensor() : tensor<3072x120xf32>
      %12 = bufferization.alloc_tensor() : tensor<256x120xf32>
      %13 = scf.forall (%arg0, %arg1) in (128, 60) shared_outs(%arg2 = %12) -> (tensor<256x120xf32>) {
        %16 = affine.apply #map(%arg0)
        %17 = affine.apply #map(%arg1)
        %18 = affine.apply #map(%arg0)
        %19 = affine.apply #map(%arg1)
        %extracted_slice = tensor.extract_slice %10[0, %16] [3072, 2] [1, 1] : tensor<3072x256xf32> to tensor<3072x2xf32>
        %extracted_slice_0 = tensor.extract_slice %11[0, %17] [3072, 2] [1, 1] : tensor<3072x120xf32> to tensor<3072x2xf32>
        %extracted_slice_1 = tensor.extract_slice %arg2[%18, %19] [2, 2] [1, 1] : tensor<256x120xf32> to tensor<2x2xf32>
        %20 = scf.for %arg3 = %c0 to %c3072 step %c2 iter_args(%arg4 = %extracted_slice_1) -> (tensor<2x2xf32>) {
          %extracted_slice_2 = tensor.extract_slice %extracted_slice[%arg3, 0] [2, 2] [1, 1] : tensor<3072x2xf32> to tensor<2x2xf32>
          %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg3, 0] [2, 2] [1, 1] : tensor<3072x2xf32> to tensor<2x2xf32>
          %23 = linalg.matmul_transpose_a {tag = "operation_5"} ins(%extracted_slice_2, %extracted_slice_3 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%arg4 : tensor<2x2xf32>) -> tensor<2x2xf32>
          scf.yield %23 : tensor<2x2xf32>
        }
        %21 = affine.apply #map(%arg0)
        %22 = affine.apply #map(%arg1)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %20 into %arg2[%21, %22] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<256x120xf32>
        }
      }
      %14 = call @nanoTime() : () -> i64
      %15 = arith.subi %14, %6 : i64
      call @printI64(%15) : (i64) -> ()
      call @printNewline() : () -> ()
      return %13 : tensor<256x120xf32>
    }
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @matmul() : () -> tensor<256x120xf32>
      }
      return
    }
  }
  
