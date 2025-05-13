
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<224x15xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<1536x512xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<1536x512xf32>) -> tensor<1536x512xf32>
    %2 = bufferization.alloc_tensor() : tensor<1536x512xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<1536x512xf32>) -> tensor<1536x512xf32>
    %4 = call @nanoTime() : () -> i64
    %5 = bufferization.alloc_tensor() : tensor<1536xf32>
    %6 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%5 : tensor<1536xf32>) -> tensor<1536xf32>
    %reduced = linalg.reduce ins(%1 : tensor<1536x512xf32>) outs(%6 : tensor<1536xf32>) dimensions = [1]  {tag = "operation_3"}
      (%in: f32, %init: f32) {
        %22 = arith.maximumf %in, %init : f32
        linalg.yield %22 : f32
      }
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "none"} ins(%1, %reduced : tensor<1536x512xf32>, tensor<1536xf32>) outs(%3 : tensor<1536x512xf32>) attrs =  {tag = "operation_4"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %22 = arith.subf %in, %in_1 : f32
      %23 = math.exp %22 : f32
      linalg.yield %23 : f32
    } -> tensor<1536x512xf32>
    %8 = bufferization.alloc_tensor() : tensor<512x256xf32>
    %9 = bufferization.alloc_tensor() : tensor<1536x256xf32>
    %10 = linalg.matmul {tag = "operation_5"} ins(%7, %8 : tensor<1536x512xf32>, tensor<512x256xf32>) outs(%9 : tensor<1536x256xf32>) -> tensor<1536x256xf32>
    %11 = bufferization.alloc_tensor() : tensor<1536x224xf32>
    %12 = bufferization.alloc_tensor() : tensor<256x224xf32>
    %13 = linalg.matmul_transpose_a {tag = "operation_6"} ins(%10, %11 : tensor<1536x256xf32>, tensor<1536x224xf32>) outs(%12 : tensor<256x224xf32>) -> tensor<256x224xf32>
    %14 = bufferization.alloc_tensor() : tensor<256x224xf32>
    %15 = bufferization.alloc_tensor() : tensor<256x224xf32>
    %16 = linalg.add {tag = "operation_7"} ins(%13, %14 : tensor<256x224xf32>, tensor<256x224xf32>) outs(%15 : tensor<256x224xf32>) -> tensor<256x224xf32>
    %17 = bufferization.alloc_tensor() : tensor<256x15xf32>
    %18 = bufferization.alloc_tensor() : tensor<224x15xf32>
    %19 = linalg.matmul_transpose_a {tag = "operation_8"} ins(%16, %17 : tensor<256x224xf32>, tensor<256x15xf32>) outs(%18 : tensor<224x15xf32>) -> tensor<224x15xf32>
    %20 = call @nanoTime() : () -> i64
    %21 = arith.subi %20, %4 : i64
    call @printI64(%21) : (i64) -> ()
    call @printNewline() : () -> ()
    return %19 : tensor<224x15xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<224x15xf32>
    }
    return
  }
}
