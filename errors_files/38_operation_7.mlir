
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<128x21x11x288xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<128x48x56x56xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<128x48x56x56xf32>) -> tensor<128x48x56x56xf32>
    %2 = bufferization.alloc_tensor() : tensor<48x48x3x3xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
    %4 = bufferization.alloc_tensor() : tensor<128x48x27x27xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<128x48x27x27xf32>) -> tensor<128x48x27x27xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_3"} ins(%1, %3 : tensor<128x48x56x56xf32>, tensor<48x48x3x3xf32>) outs(%5 : tensor<128x48x27x27xf32>) -> tensor<128x48x27x27xf32>
    %8 = bufferization.alloc_tensor() : tensor<1x1xf32>
    %9 = bufferization.alloc_tensor() : tensor<128x48x27x27xf32>
    %10 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_4"} ins(%7, %8 : tensor<128x48x27x27xf32>, tensor<1x1xf32>) outs(%9 : tensor<128x48x27x27xf32>) -> tensor<128x48x27x27xf32>
    %11 = bufferization.alloc_tensor() : tensor<7x7xf32>
    %12 = bufferization.alloc_tensor() : tensor<128x42x21x27xf32>
    %13 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_5"} ins(%10, %11 : tensor<128x48x27x27xf32>, tensor<7x7xf32>) outs(%12 : tensor<128x42x21x27xf32>) -> tensor<128x42x21x27xf32>
    %14 = bufferization.alloc_tensor() : tensor<1x1xf32>
    %15 = bufferization.alloc_tensor() : tensor<128x42x21x27xf32>
    %16 = linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_6"} ins(%13, %14 : tensor<128x42x21x27xf32>, tensor<1x1xf32>) outs(%15 : tensor<128x42x21x27xf32>) -> tensor<128x42x21x27xf32>
    %17 = bufferization.alloc_tensor() : tensor<1x1x27x288xf32>
    %18 = bufferization.alloc_tensor() : tensor<128x21x11x288xf32>
    %19 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_7"} ins(%16, %17 : tensor<128x42x21x27xf32>, tensor<1x1x27x288xf32>) outs(%18 : tensor<128x21x11x288xf32>) -> tensor<128x21x11x288xf32>
    %20 = call @nanoTime() : () -> i64
    %21 = arith.subi %20, %6 : i64
    call @printI64(%21) : (i64) -> ()
    call @printNewline() : () -> ()
    return %19 : tensor<128x21x11x288xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<128x21x11x288xf32>
    }
    return
  }
}
