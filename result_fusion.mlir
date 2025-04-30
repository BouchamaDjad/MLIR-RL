module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @matmul() -> memref<256x120x2x3xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512x15x15xf32>
    linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%alloc : memref<256x512x15x15xf32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<240x512x7x7xf32>
    linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%alloc_0 : memref<240x512x7x7xf32>)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x240x5x5xf32>
    linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%alloc_1 : memref<256x240x5x5xf32>)
    %0 = call @nanoTime() : () -> i64
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_3"} ins(%alloc, %alloc_0 : memref<256x512x15x15xf32>, memref<240x512x7x7xf32>) outs(%alloc_1 : memref<256x240x5x5xf32>)
    memref.dealloc %alloc_0 : memref<240x512x7x7xf32>
    memref.dealloc %alloc : memref<256x512x15x15xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256x240x3x3xf32>
    linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_4"} ins(%alloc_1, %alloc_2 : memref<256x240x5x5xf32>, memref<3x3xf32>) outs(%alloc_3 : memref<256x240x3x3xf32>)
    memref.dealloc %alloc_2 : memref<3x3xf32>
    memref.dealloc %alloc_1 : memref<256x240x5x5xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<256x120x2x3xf32>
    linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_5"} ins(%alloc_3, %alloc_4 : memref<256x240x3x3xf32>, memref<1x1xf32>) outs(%alloc_5 : memref<256x120x2x3xf32>)
    memref.dealloc %alloc_4 : memref<1x1xf32>
    memref.dealloc %alloc_3 : memref<256x240x3x3xf32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<256x120x2x3xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<256x120x2x3xf32>
    linalg.add {tag = "operation_6"} ins(%alloc_5, %alloc_6 : memref<256x120x2x3xf32>, memref<256x120x2x3xf32>) outs(%alloc_7 : memref<256x120x2x3xf32>)
    memref.dealloc %alloc_6 : memref<256x120x2x3xf32>
    memref.dealloc %alloc_5 : memref<256x120x2x3xf32>
    %1 = call @nanoTime() : () -> i64
    %2 = arith.subi %1, %0 : i64
    call @printI64(%2) : (i64) -> ()
    call @printNewline() : () -> ()
    return %alloc_7 : memref<256x120x2x3xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @nanoTime() : () -> i64
      %1 = func.call @nanoTime() : () -> i64
      %2 = arith.subi %1, %0 : i64
      func.call @printI64(%2) : (i64) -> ()
      func.call @printNewline() : () -> ()
    }
    return
  }
}

