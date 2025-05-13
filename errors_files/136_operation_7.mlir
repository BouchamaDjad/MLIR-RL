
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<768x130xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<768x512xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<768x512xf32>) -> tensor<768x512xf32>
    %2 = bufferization.alloc_tensor() : tensor<512x1536xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<512x1536xf32>) -> tensor<512x1536xf32>
    %4 = bufferization.alloc_tensor() : tensor<768x1536xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<768x1536xf32>) -> tensor<768x1536xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.matmul {tag = "operation_3"} ins(%1, %3 : tensor<768x512xf32>, tensor<512x1536xf32>) outs(%5 : tensor<768x1536xf32>) -> tensor<768x1536xf32>
    %8 = bufferization.alloc_tensor() : tensor<130x1536xf32>
    %9 = bufferization.alloc_tensor() : tensor<768x130xf32>
    %10 = linalg.matmul_transpose_b {tag = "operation_4"} ins(%7, %8 : tensor<768x1536xf32>, tensor<130x1536xf32>) outs(%9 : tensor<768x130xf32>) -> tensor<768x130xf32>
    %11 = bufferization.alloc_tensor() : tensor<768x130xf32>
    %12 = bufferization.alloc_tensor() : tensor<768xf32>
    %13 = linalg.fill {tag = "operation_5"} ins(%cst : f32) outs(%12 : tensor<768xf32>) -> tensor<768xf32>
    %reduced = linalg.reduce ins(%10 : tensor<768x130xf32>) outs(%13 : tensor<768xf32>) dimensions = [1]  {tag = "operation_6"}
      (%in: f32, %init: f32) {
        %17 = arith.maximumf %in, %init : f32
        linalg.yield %17 : f32
      }
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "none"} ins(%10, %reduced : tensor<768x130xf32>, tensor<768xf32>) outs(%11 : tensor<768x130xf32>) attrs =  {tag = "operation_7"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %17 = arith.subf %in, %in_1 : f32
      %18 = math.exp %17 : f32
      linalg.yield %18 : f32
    } -> tensor<768x130xf32>
    %15 = call @nanoTime() : () -> i64
    %16 = arith.subi %15, %6 : i64
    call @printI64(%16) : (i64) -> ()
    call @printNewline() : () -> ()
    return %14 : tensor<768x130xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<768x130xf32>
    }
    return
  }
}
