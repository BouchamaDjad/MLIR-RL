
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<2048x1536xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<2048x1536xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<2048x1536xf32>) -> tensor<2048x1536xf32>
    %2 = bufferization.alloc_tensor() : tensor<1536x3072xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<1536x3072xf32>) -> tensor<1536x3072xf32>
    %4 = bufferization.alloc_tensor() : tensor<2048x3072xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<2048x3072xf32>) -> tensor<2048x3072xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.matmul {tag = "operation_3"} ins(%1, %3 : tensor<2048x1536xf32>, tensor<1536x3072xf32>) outs(%5 : tensor<2048x3072xf32>) -> tensor<2048x3072xf32>
    %8 = bufferization.alloc_tensor() : tensor<3072x1536xf32>
    %9 = bufferization.alloc_tensor() : tensor<2048x1536xf32>
    %10 = linalg.matmul {tag = "operation_4"} ins(%7, %8 : tensor<2048x3072xf32>, tensor<3072x1536xf32>) outs(%9 : tensor<2048x1536xf32>) -> tensor<2048x1536xf32>
    %11 = call @nanoTime() : () -> i64
    %12 = arith.subi %11, %6 : i64
    call @printI64(%12) : (i64) -> ()
    call @printNewline() : () -> ()
    return %10 : tensor<2048x1536xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<2048x1536xf32>
    }
    return
  }
}
