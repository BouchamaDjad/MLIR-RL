
#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<3072x15xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<3072x128xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<3072x128xf32>) -> tensor<3072x128xf32>
    %2 = bufferization.alloc_tensor() : tensor<15x128xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<15x128xf32>) -> tensor<15x128xf32>
    %4 = bufferization.alloc_tensor() : tensor<3072x15xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<3072x15xf32>) -> tensor<3072x15xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.matmul_transpose_b {tag = "operation_3"} ins(%1, %3 : tensor<3072x128xf32>, tensor<15x128xf32>) outs(%5 : tensor<3072x15xf32>) -> tensor<3072x15xf32>
    %8 = bufferization.alloc_tensor() : tensor<3072x15xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<3072x15xf32>) outs(%8 : tensor<3072x15xf32>) attrs =  {tag = "operation_4"} {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.negf %in : f32
      %13 = math.exp %12 : f32
      %14 = arith.addf %13, %cst : f32
      %15 = arith.divf %cst, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<3072x15xf32>
    %10 = call @nanoTime() : () -> i64
    %11 = arith.subi %10, %6 : i64
    call @printI64(%11) : (i64) -> ()
    call @printNewline() : () -> ()
    return %9 : tensor<3072x15xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<3072x15xf32>
    }
    return
  }
}
