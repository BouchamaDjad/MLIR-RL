
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<256x30x50x240xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<256x150x228x240xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<256x150x228x240xf32>) -> tensor<256x150x228x240xf32>
    %2 = bufferization.alloc_tensor() : tensor<7x7xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<7x7xf32>) -> tensor<7x7xf32>
    %4 = bufferization.alloc_tensor() : tensor<256x72x111x240xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<256x72x111x240xf32>) -> tensor<256x72x111x240xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_3"} ins(%1, %3 : tensor<256x150x228x240xf32>, tensor<7x7xf32>) outs(%5 : tensor<256x72x111x240xf32>) -> tensor<256x72x111x240xf32>
    %8 = bufferization.alloc_tensor() : tensor<1x1xf32>
    %9 = bufferization.alloc_tensor() : tensor<256x36x56x240xf32>
    %10 = linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_4"} ins(%7, %8 : tensor<256x72x111x240xf32>, tensor<1x1xf32>) outs(%9 : tensor<256x36x56x240xf32>) -> tensor<256x36x56x240xf32>
    %11 = bufferization.alloc_tensor() : tensor<7x7xf32>
    %12 = bufferization.alloc_tensor() : tensor<256x30x50x240xf32>
    %13 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_5"} ins(%10, %11 : tensor<256x36x56x240xf32>, tensor<7x7xf32>) outs(%12 : tensor<256x30x50x240xf32>) -> tensor<256x30x50x240xf32>
    %14 = call @nanoTime() : () -> i64
    %15 = arith.subi %14, %6 : i64
    call @printI64(%15) : (i64) -> ()
    call @printNewline() : () -> ()
    return %13 : tensor<256x30x50x240xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<256x30x50x240xf32>
    }
    return
  }
}
