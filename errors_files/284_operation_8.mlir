
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<128x4x56x512xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<128x7x224x64xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<128x7x224x64xf32>) -> tensor<128x7x224x64xf32>
    %2 = bufferization.alloc_tensor() : tensor<1x1xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %4 = bufferization.alloc_tensor() : tensor<128x7x224x64xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<128x7x224x64xf32>) -> tensor<128x7x224x64xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_3"} ins(%1, %3 : tensor<128x7x224x64xf32>, tensor<1x1xf32>) outs(%5 : tensor<128x7x224x64xf32>) -> tensor<128x7x224x64xf32>
    %8 = bufferization.alloc_tensor() : tensor<3x3xf32>
    %9 = bufferization.alloc_tensor() : tensor<128x7x111x31xf32>
    %10 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_4"} ins(%7, %8 : tensor<128x7x224x64xf32>, tensor<3x3xf32>) outs(%9 : tensor<128x7x111x31xf32>) -> tensor<128x7x111x31xf32>
    %11 = bufferization.alloc_tensor() : tensor<128x7x111x31xf32>
    %12 = bufferization.alloc_tensor() : tensor<128x7x111xf32>
    %13 = linalg.fill {tag = "operation_5"} ins(%cst : f32) outs(%12 : tensor<128x7x111xf32>) -> tensor<128x7x111xf32>
    %reduced = linalg.reduce ins(%10 : tensor<128x7x111x31xf32>) outs(%13 : tensor<128x7x111xf32>) dimensions = [3]  {tag = "operation_6"}
      (%in: f32, %init: f32) {
        %23 = arith.maximumf %in, %init : f32
        linalg.yield %23 : f32
      }
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = "none"} ins(%10, %reduced : tensor<128x7x111x31xf32>, tensor<128x7x111xf32>) outs(%11 : tensor<128x7x111x31xf32>) attrs =  {tag = "operation_7"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %23 = arith.subf %in, %in_1 : f32
      %24 = math.exp %23 : f32
      linalg.yield %24 : f32
    } -> tensor<128x7x111x31xf32>
    %15 = bufferization.alloc_tensor() : tensor<1x1x31x512xf32>
    %16 = bufferization.alloc_tensor() : tensor<128x4x56x512xf32>
    %17 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_8"} ins(%14, %15 : tensor<128x7x111x31xf32>, tensor<1x1x31x512xf32>) outs(%16 : tensor<128x4x56x512xf32>) -> tensor<128x4x56x512xf32>
    %18 = bufferization.alloc_tensor() : tensor<1x1xf32>
    %19 = bufferization.alloc_tensor() : tensor<128x4x56x512xf32>
    %20 = linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_9"} ins(%17, %18 : tensor<128x4x56x512xf32>, tensor<1x1xf32>) outs(%19 : tensor<128x4x56x512xf32>) -> tensor<128x4x56x512xf32>
    %21 = call @nanoTime() : () -> i64
    %22 = arith.subi %21, %6 : i64
    call @printI64(%22) : (i64) -> ()
    call @printNewline() : () -> ()
    return %20 : tensor<128x4x56x512xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<128x4x56x512xf32>
    }
    return
  }
}
