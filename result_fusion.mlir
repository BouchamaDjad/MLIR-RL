#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 2)>
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
    %0 = call @nanoTime() : () -> i64
    scf.forall (%arg0, %arg1) in (64, 120) {
      %3 = affine.apply #map(%arg0)
      %4 = affine.apply #map1(%arg1)
      %subview = memref.subview %alloc[%3, 0, 0, 0] [4, 512, 15, 15] [1, 1, 1, 1] : memref<256x512x15x15xf32> to memref<4x512x15x15xf32, strided<[115200, 225, 15, 1], offset: ?>>
      %subview_8 = memref.subview %alloc_0[%4, 0, 0, 0] [2, 512, 7, 7] [1, 1, 1, 1] : memref<240x512x7x7xf32> to memref<2x512x7x7xf32, strided<[25088, 49, 7, 1], offset: ?>>
      %5 = affine.apply #map(%arg0)
      %6 = affine.apply #map1(%arg1)
      %subview_9 = memref.subview %alloc_1[%5, %6, 0, 0] [4, 2, 5, 5] [1, 1, 1, 1] : memref<256x240x5x5xf32> to memref<4x2x5x5xf32, strided<[6000, 25, 5, 1], offset: ?>>
      linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%subview_9 : memref<4x2x5x5xf32, strided<[6000, 25, 5, 1], offset: ?>>)
      linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_3"} ins(%subview, %subview_8 : memref<4x512x15x15xf32, strided<[115200, 225, 15, 1], offset: ?>>, memref<2x512x7x7xf32, strided<[25088, 49, 7, 1], offset: ?>>) outs(%subview_9 : memref<4x2x5x5xf32, strided<[6000, 25, 5, 1], offset: ?>>)
      %7 = affine.apply #map(%arg0)
      %8 = affine.apply #map1(%arg1)
      %subview_10 = memref.subview %alloc_1[%7, %8, 0, 0] [4, 2, 5, 5] [1, 1, 1, 1] : memref<256x240x5x5xf32> to memref<4x2x5x5xf32, strided<[6000, 25, 5, 1], offset: ?>>
      memref.copy %subview_9, %subview_10 : memref<4x2x5x5xf32, strided<[6000, 25, 5, 1], offset: ?>> to memref<4x2x5x5xf32, strided<[6000, 25, 5, 1], offset: ?>>
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256x240x3x3xf32>
    linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_4"} ins(%alloc_1, %alloc_2 : memref<256x240x5x5xf32>, memref<3x3xf32>) outs(%alloc_3 : memref<256x240x3x3xf32>)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<256x120x2x3xf32>
    linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_5"} ins(%alloc_3, %alloc_4 : memref<256x240x3x3xf32>, memref<1x1xf32>) outs(%alloc_5 : memref<256x120x2x3xf32>)
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<256x120x2x3xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<256x120x2x3xf32>
    linalg.add {tag = "operation_6"} ins(%alloc_5, %alloc_6 : memref<256x120x2x3xf32>, memref<256x120x2x3xf32>) outs(%alloc_7 : memref<256x120x2x3xf32>)
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

