
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0 * 4)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<256x7xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<256x2048xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %2 = bufferization.alloc_tensor() : tensor<256x2048xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %4 = call @nanoTime() : () -> i64
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<256x2048xf32>) outs(%3 : tensor<256x2048xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.negf %in : f32
      %15 = math.exp %14 : f32
      %16 = arith.addf %15, %cst : f32
      %17 = arith.divf %cst, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<256x2048xf32>
    %6 = bufferization.alloc_tensor() : tensor<2048x3072xf32>
    %7 = bufferization.alloc_tensor() : tensor<256x3072xf32>
    %8 = linalg.matmul {tag = "operation_3"} ins(%5, %6 : tensor<256x2048xf32>, tensor<2048x3072xf32>) outs(%7 : tensor<256x3072xf32>) -> tensor<256x3072xf32>
    %9 = bufferization.alloc_tensor() : tensor<7x3072xf32>
    %10 = bufferization.alloc_tensor() : tensor<256x7xf32>
    %11 = scf.forall (%arg0, %arg1) in (64, 7) shared_outs(%arg2 = %10) -> (tensor<256x7xf32>) {
      %14 = affine.apply #map1(%arg0)
      %15 = affine.apply #map1(%arg0)
      %extracted_slice = tensor.extract_slice %8[%14, 0] [4, 3072] [1, 1] : tensor<256x3072xf32> to tensor<4x3072xf32>
      %extracted_slice_1 = tensor.extract_slice %9[%arg1, 0] [1, 3072] [1, 1] : tensor<7x3072xf32> to tensor<1x3072xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%15, %arg1] [4, 1] [1, 1] : tensor<256x7xf32> to tensor<4x1xf32>
      %16 = linalg.matmul_transpose_b {tag = "operation_4"} ins(%extracted_slice, %extracted_slice_1 : tensor<4x3072xf32>, tensor<1x3072xf32>) outs(%extracted_slice_2 : tensor<4x1xf32>) -> tensor<4x1xf32>
      %17 = affine.apply #map1(%arg0)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %16 into %arg2[%17, %arg1] [4, 1] [1, 1] : tensor<4x1xf32> into tensor<256x7xf32>
      }
    }
    %12 = call @nanoTime() : () -> i64
    %13 = arith.subi %12, %4 : i64
    call @printI64(%13) : (i64) -> ()
    call @printNewline() : () -> ()
    return %11 : tensor<256x7xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<256x7xf32>
    }
    return
  }
}
