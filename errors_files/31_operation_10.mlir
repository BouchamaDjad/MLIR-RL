
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<120x120xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<2048x3072xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<2048x3072xf32>) -> tensor<2048x3072xf32>
    %2 = bufferization.alloc_tensor() : tensor<2048x3072xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<2048x3072xf32>) -> tensor<2048x3072xf32>
    %4 = call @nanoTime() : () -> i64
    %5 = bufferization.alloc_tensor() : tensor<2048xf32>
    %6 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%5 : tensor<2048xf32>) -> tensor<2048xf32>
    %reduced = linalg.reduce ins(%1 : tensor<2048x3072xf32>) outs(%6 : tensor<2048xf32>) dimensions = [1]  {tag = "operation_3"}
      (%in: f32, %init: f32) {
        %23 = arith.maximumf %in, %init : f32
        linalg.yield %23 : f32
      }
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "none"} ins(%1, %reduced : tensor<2048x3072xf32>, tensor<2048xf32>) outs(%3 : tensor<2048x3072xf32>) attrs =  {tag = "operation_4"} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %23 = arith.subf %in, %in_2 : f32
      %24 = math.exp %23 : f32
      linalg.yield %24 : f32
    } -> tensor<2048x3072xf32>
    %8 = bufferization.alloc_tensor() : tensor<2048x120xf32>
    %9 = bufferization.alloc_tensor() : tensor<3072x120xf32>
    %10 = linalg.matmul_transpose_a {tag = "operation_5"} ins(%7, %8 : tensor<2048x3072xf32>, tensor<2048x120xf32>) outs(%9 : tensor<3072x120xf32>) -> tensor<3072x120xf32>
    %11 = bufferization.alloc_tensor() : tensor<3072x120xf32>
    %12 = bufferization.alloc_tensor() : tensor<3072x120xf32>
    %13 = linalg.add {tag = "operation_6"} ins(%10, %11 : tensor<3072x120xf32>, tensor<3072x120xf32>) outs(%12 : tensor<3072x120xf32>) -> tensor<3072x120xf32>
    %14 = bufferization.alloc_tensor() : tensor<3072x120xf32>
    %15 = bufferization.alloc_tensor() : tensor<3072xf32>
    %16 = linalg.fill {tag = "operation_7"} ins(%cst : f32) outs(%15 : tensor<3072xf32>) -> tensor<3072xf32>
    %reduced_1 = linalg.reduce ins(%13 : tensor<3072x120xf32>) outs(%16 : tensor<3072xf32>) dimensions = [1]  {tag = "operation_8"}
      (%in: f32, %init: f32) {
        %23 = arith.maximumf %in, %init : f32
        linalg.yield %23 : f32
      }
    %17 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "none"} ins(%13, %reduced_1 : tensor<3072x120xf32>, tensor<3072xf32>) outs(%14 : tensor<3072x120xf32>) attrs =  {tag = "operation_9"} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %23 = arith.subf %in, %in_2 : f32
      %24 = math.exp %23 : f32
      linalg.yield %24 : f32
    } -> tensor<3072x120xf32>
    %18 = bufferization.alloc_tensor() : tensor<3072x120xf32>
    %19 = bufferization.alloc_tensor() : tensor<120x120xf32>
    %20 = linalg.matmul_transpose_a {tag = "operation_10"} ins(%17, %18 : tensor<3072x120xf32>, tensor<3072x120xf32>) outs(%19 : tensor<120x120xf32>) -> tensor<120x120xf32>
    %21 = call @nanoTime() : () -> i64
    %22 = arith.subi %21, %4 : i64
    call @printI64(%22) : (i64) -> ()
    call @printNewline() : () -> ()
    return %20 : tensor<120x120xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<120x120xf32>
    }
    return
  }
}
