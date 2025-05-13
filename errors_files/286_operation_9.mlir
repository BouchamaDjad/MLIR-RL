
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<128x512xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<128x512xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_1 : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %2 = bufferization.alloc_tensor() : tensor<128x512xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_1 : f32) outs(%2 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %4 = call @nanoTime() : () -> i64
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<128x512xf32>) outs(%3 : tensor<128x512xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %18 = arith.negf %in : f32
      %19 = math.exp %18 : f32
      %20 = arith.addf %19, %cst : f32
      %21 = arith.divf %cst, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<128x512xf32>
    %6 = bufferization.alloc_tensor() : tensor<128x512xf32>
    %7 = bufferization.alloc_tensor() : tensor<128xf32>
    %8 = linalg.fill {tag = "operation_3"} ins(%cst_0 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    %reduced = linalg.reduce ins(%5 : tensor<128x512xf32>) outs(%8 : tensor<128xf32>) dimensions = [1]  {tag = "operation_4"}
      (%in: f32, %init: f32) {
        %18 = arith.maximumf %in, %init : f32
        linalg.yield %18 : f32
      }
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "none"} ins(%5, %reduced : tensor<128x512xf32>, tensor<128xf32>) outs(%6 : tensor<128x512xf32>) attrs =  {tag = "operation_5"} {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %18 = arith.subf %in, %in_3 : f32
      %19 = math.exp %18 : f32
      linalg.yield %19 : f32
    } -> tensor<128x512xf32>
    %10 = bufferization.alloc_tensor() : tensor<128x512xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<128x512xf32>) outs(%10 : tensor<128x512xf32>) attrs =  {tag = "operation_6"} {
    ^bb0(%in: f32, %out: f32):
      %18 = arith.negf %in : f32
      %19 = math.exp %18 : f32
      %20 = arith.addf %19, %cst : f32
      %21 = arith.divf %cst, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<128x512xf32>
    %12 = bufferization.alloc_tensor() : tensor<128x512xf32>
    %13 = bufferization.alloc_tensor() : tensor<128xf32>
    %14 = linalg.fill {tag = "operation_7"} ins(%cst_0 : f32) outs(%13 : tensor<128xf32>) -> tensor<128xf32>
    %reduced_2 = linalg.reduce ins(%11 : tensor<128x512xf32>) outs(%14 : tensor<128xf32>) dimensions = [1]  {tag = "operation_8"}
      (%in: f32, %init: f32) {
        %18 = arith.maximumf %in, %init : f32
        linalg.yield %18 : f32
      }
    %15 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "none"} ins(%11, %reduced_2 : tensor<128x512xf32>, tensor<128xf32>) outs(%12 : tensor<128x512xf32>) attrs =  {tag = "operation_9"} {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %18 = arith.subf %in, %in_3 : f32
      %19 = math.exp %18 : f32
      linalg.yield %19 : f32
    } -> tensor<128x512xf32>
    %16 = call @nanoTime() : () -> i64
    %17 = arith.subi %16, %4 : i64
    call @printI64(%17) : (i64) -> ()
    call @printNewline() : () -> ()
    return %15 : tensor<128x512xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<128x512xf32>
    }
    return
  }
}
