
#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<228x130xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<2048x1024xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    %2 = bufferization.alloc_tensor() : tensor<228x1024xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<228x1024xf32>) -> tensor<228x1024xf32>
    %4 = bufferization.alloc_tensor() : tensor<2048x228xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<2048x228xf32>) -> tensor<2048x228xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.matmul_transpose_b {tag = "operation_3"} ins(%1, %3 : tensor<2048x1024xf32>, tensor<228x1024xf32>) outs(%5 : tensor<2048x228xf32>) -> tensor<2048x228xf32>
    %8 = bufferization.alloc_tensor() : tensor<2048x228xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<2048x228xf32>) outs(%8 : tensor<2048x228xf32>) attrs =  {tag = "operation_4"} {
    ^bb0(%in: f32, %out: f32):
      %18 = arith.negf %in : f32
      %19 = math.exp %18 : f32
      %20 = arith.addf %19, %cst : f32
      %21 = arith.divf %cst, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<2048x228xf32>
    %10 = bufferization.alloc_tensor() : tensor<2048x130xf32>
    %11 = bufferization.alloc_tensor() : tensor<228x130xf32>
    %12 = linalg.matmul_transpose_a {tag = "operation_5"} ins(%9, %10 : tensor<2048x228xf32>, tensor<2048x130xf32>) outs(%11 : tensor<228x130xf32>) -> tensor<228x130xf32>
    %13 = bufferization.alloc_tensor() : tensor<228x130xf32>
    %14 = bufferization.alloc_tensor() : tensor<228x130xf32>
    %15 = linalg.add {tag = "operation_6"} ins(%12, %13 : tensor<228x130xf32>, tensor<228x130xf32>) outs(%14 : tensor<228x130xf32>) -> tensor<228x130xf32>
    %16 = call @nanoTime() : () -> i64
    %17 = arith.subi %16, %6 : i64
    call @printI64(%17) : (i64) -> ()
    call @printNewline() : () -> ()
    return %15 : tensor<228x130xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<228x130xf32>
    }
    return
  }
}
