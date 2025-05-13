
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<256x106x240xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<256x112x240xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<256x112x240xf32>) -> tensor<256x112x240xf32>
    %2 = bufferization.alloc_tensor() : tensor<7xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<7xf32>) -> tensor<7xf32>
    %4 = bufferization.alloc_tensor() : tensor<256x106x240xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<256x106x240xf32>) -> tensor<256x106x240xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.pooling_nwc_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>, tag = "operation_3"} ins(%1, %3 : tensor<256x112x240xf32>, tensor<7xf32>) outs(%5 : tensor<256x106x240xf32>) -> tensor<256x106x240xf32>
    %8 = bufferization.alloc_tensor() : tensor<256x106x240xf32>
    %9 = bufferization.alloc_tensor() : tensor<256x106x240xf32>
    %10 = linalg.add {tag = "operation_4"} ins(%7, %8 : tensor<256x106x240xf32>, tensor<256x106x240xf32>) outs(%9 : tensor<256x106x240xf32>) -> tensor<256x106x240xf32>
    %11 = bufferization.alloc_tensor() : tensor<256x106x240xf32>
    %12 = bufferization.alloc_tensor() : tensor<256x106xf32>
    %13 = linalg.fill {tag = "operation_5"} ins(%cst : f32) outs(%12 : tensor<256x106xf32>) -> tensor<256x106xf32>
    %reduced = linalg.reduce ins(%10 : tensor<256x106x240xf32>) outs(%13 : tensor<256x106xf32>) dimensions = [2]  {tag = "operation_6"}
      (%in: f32, %init: f32) {
        %21 = arith.maximumf %in, %init : f32
        linalg.yield %21 : f32
      }
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = "none"} ins(%10, %reduced : tensor<256x106x240xf32>, tensor<256x106xf32>) outs(%11 : tensor<256x106x240xf32>) attrs =  {tag = "operation_7"} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %21 = arith.subf %in, %in_2 : f32
      %22 = math.exp %21 : f32
      linalg.yield %22 : f32
    } -> tensor<256x106x240xf32>
    %15 = bufferization.alloc_tensor() : tensor<256x106x240xf32>
    %16 = bufferization.alloc_tensor() : tensor<256x106xf32>
    %17 = linalg.fill {tag = "operation_8"} ins(%cst : f32) outs(%16 : tensor<256x106xf32>) -> tensor<256x106xf32>
    %reduced_1 = linalg.reduce ins(%14 : tensor<256x106x240xf32>) outs(%17 : tensor<256x106xf32>) dimensions = [2]  {tag = "operation_9"}
      (%in: f32, %init: f32) {
        %21 = arith.maximumf %in, %init : f32
        linalg.yield %21 : f32
      }
    %18 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = "none"} ins(%14, %reduced_1 : tensor<256x106x240xf32>, tensor<256x106xf32>) outs(%15 : tensor<256x106x240xf32>) attrs =  {tag = "operation_10"} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %21 = arith.subf %in, %in_2 : f32
      %22 = math.exp %21 : f32
      linalg.yield %22 : f32
    } -> tensor<256x106x240xf32>
    %19 = call @nanoTime() : () -> i64
    %20 = arith.subi %19, %6 : i64
    call @printI64(%20) : (i64) -> ()
    call @printNewline() : () -> ()
    return %18 : tensor<256x106x240xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<256x106x240xf32>
    }
    return
  }
}
