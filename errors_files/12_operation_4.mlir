
#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<2048x112xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<128x2048xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<128x2048xf32>) -> tensor<128x2048xf32>
    %2 = bufferization.alloc_tensor() : tensor<128x2048xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<128x2048xf32>) -> tensor<128x2048xf32>
    %4 = call @nanoTime() : () -> i64
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<128x2048xf32>) outs(%3 : tensor<128x2048xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.negf %in : f32
      %15 = math.exp %14 : f32
      %16 = arith.addf %15, %cst : f32
      %17 = arith.divf %cst, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x2048xf32>
    %6 = bufferization.alloc_tensor() : tensor<128x224xf32>
    %7 = bufferization.alloc_tensor() : tensor<2048x224xf32>
    %8 = linalg.matmul_transpose_a {tag = "operation_3"} ins(%5, %6 : tensor<128x2048xf32>, tensor<128x224xf32>) outs(%7 : tensor<2048x224xf32>) -> tensor<2048x224xf32>
    %9 = bufferization.alloc_tensor() : tensor<112x224xf32>
    %10 = bufferization.alloc_tensor() : tensor<2048x112xf32>
    %11 = linalg.matmul_transpose_b {tag = "operation_4"} ins(%8, %9 : tensor<2048x224xf32>, tensor<112x224xf32>) outs(%10 : tensor<2048x112xf32>) -> tensor<2048x112xf32>
    %12 = call @nanoTime() : () -> i64
    %13 = arith.subi %12, %4 : i64
    call @printI64(%13) : (i64) -> ()
    call @printNewline() : () -> ()
    return %11 : tensor<2048x112xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<2048x112xf32>
    }
    return
  }
}
