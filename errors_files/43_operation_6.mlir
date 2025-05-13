
#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<3072x3072xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<256x3072xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<256x3072xf32>) -> tensor<256x3072xf32>
    %2 = bufferization.alloc_tensor() : tensor<256x3072xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<256x3072xf32>) -> tensor<256x3072xf32>
    %4 = call @nanoTime() : () -> i64
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<256x3072xf32>) outs(%3 : tensor<256x3072xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %20 = arith.negf %in : f32
      %21 = math.exp %20 : f32
      %22 = arith.addf %21, %cst : f32
      %23 = arith.divf %cst, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<256x3072xf32>
    %6 = bufferization.alloc_tensor() : tensor<256x7xf32>
    %7 = bufferization.alloc_tensor() : tensor<3072x7xf32>
    %8 = linalg.matmul_transpose_a {tag = "operation_3"} ins(%5, %6 : tensor<256x3072xf32>, tensor<256x7xf32>) outs(%7 : tensor<3072x7xf32>) -> tensor<3072x7xf32>
    %9 = bufferization.alloc_tensor() : tensor<7x1536xf32>
    %10 = bufferization.alloc_tensor() : tensor<3072x1536xf32>
    %11 = linalg.matmul {tag = "operation_4"} ins(%8, %9 : tensor<3072x7xf32>, tensor<7x1536xf32>) outs(%10 : tensor<3072x1536xf32>) -> tensor<3072x1536xf32>
    %12 = bufferization.alloc_tensor() : tensor<28x1536xf32>
    %13 = bufferization.alloc_tensor() : tensor<3072x28xf32>
    %14 = linalg.matmul_transpose_b {tag = "operation_5"} ins(%11, %12 : tensor<3072x1536xf32>, tensor<28x1536xf32>) outs(%13 : tensor<3072x28xf32>) -> tensor<3072x28xf32>
    %15 = bufferization.alloc_tensor() : tensor<28x3072xf32>
    %16 = bufferization.alloc_tensor() : tensor<3072x3072xf32>
    %17 = linalg.matmul {tag = "operation_6"} ins(%14, %15 : tensor<3072x28xf32>, tensor<28x3072xf32>) outs(%16 : tensor<3072x3072xf32>) -> tensor<3072x3072xf32>
    %18 = call @nanoTime() : () -> i64
    %19 = arith.subi %18, %4 : i64
    call @printI64(%19) : (i64) -> ()
    call @printNewline() : () -> ()
    return %17 : tensor<3072x3072xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<3072x3072xf32>
    }
    return
  }
}
