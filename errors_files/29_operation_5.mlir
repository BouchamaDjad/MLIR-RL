
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<128x2048xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<128x2048xf32>) -> tensor<128x2048xf32>
    %2 = bufferization.alloc_tensor() : tensor<128x2048xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<128x2048xf32>) -> tensor<128x2048xf32>
    %4 = call @nanoTime() : () -> i64
    %5 = bufferization.alloc_tensor() : tensor<128xf32>
    %6 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
    %reduced = linalg.reduce ins(%1 : tensor<128x2048xf32>) outs(%6 : tensor<128xf32>) dimensions = [1]  {tag = "operation_3"}
      (%in: f32, %init: f32) {
        %13 = arith.maximumf %in, %init : f32
        linalg.yield %13 : f32
      }
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "none"} ins(%1, %reduced : tensor<128x2048xf32>, tensor<128xf32>) outs(%3 : tensor<128x2048xf32>) attrs =  {tag = "operation_4"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %13 = arith.subf %in, %in_1 : f32
      %14 = math.exp %13 : f32
      linalg.yield %14 : f32
    } -> tensor<128x2048xf32>
    %8 = bufferization.alloc_tensor() : tensor<2048x128xf32>
    %9 = bufferization.alloc_tensor() : tensor<128x128xf32>
    %10 = linalg.matmul {tag = "operation_5"} ins(%7, %8 : tensor<128x2048xf32>, tensor<2048x128xf32>) outs(%9 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %11 = call @nanoTime() : () -> i64
    %12 = arith.subi %11, %4 : i64
    call @printI64(%12) : (i64) -> ()
    call @printNewline() : () -> ()
    return %10 : tensor<128x128xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<128x128xf32>
    }
    return
  }
}
