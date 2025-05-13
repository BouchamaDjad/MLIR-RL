
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<256x512xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<256x256xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_1 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %2 = bufferization.alloc_tensor() : tensor<256x256xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_1 : f32) outs(%2 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %4 = call @nanoTime() : () -> i64
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<256x256xf32>) outs(%3 : tensor<256x256xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.negf %in : f32
      %16 = math.exp %15 : f32
      %17 = arith.addf %16, %cst : f32
      %18 = arith.divf %cst, %17 : f32
      linalg.yield %18 : f32
    } -> tensor<256x256xf32>
    %6 = bufferization.alloc_tensor() : tensor<256x512xf32>
    %7 = bufferization.alloc_tensor() : tensor<256x512xf32>
    %8 = linalg.matmul {tag = "operation_3"} ins(%5, %6 : tensor<256x256xf32>, tensor<256x512xf32>) outs(%7 : tensor<256x512xf32>) -> tensor<256x512xf32>
    %9 = bufferization.alloc_tensor() : tensor<256x512xf32>
    %10 = bufferization.alloc_tensor() : tensor<256xf32>
    %11 = linalg.fill {tag = "operation_4"} ins(%cst_0 : f32) outs(%10 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%8 : tensor<256x512xf32>) outs(%11 : tensor<256xf32>) dimensions = [1]  {tag = "operation_5"}
      (%in: f32, %init: f32) {
        %15 = arith.maximumf %in, %init : f32
        linalg.yield %15 : f32
      }
    %12 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "none"} ins(%8, %reduced : tensor<256x512xf32>, tensor<256xf32>) outs(%9 : tensor<256x512xf32>) attrs =  {tag = "operation_6"} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %15 = arith.subf %in, %in_2 : f32
      %16 = math.exp %15 : f32
      linalg.yield %16 : f32
    } -> tensor<256x512xf32>
    %13 = call @nanoTime() : () -> i64
    %14 = arith.subi %13, %4 : i64
    call @printI64(%14) : (i64) -> ()
    call @printNewline() : () -> ()
    return %12 : tensor<256x512xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<256x512xf32>
    }
    return
  }
}
