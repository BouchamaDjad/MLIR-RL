
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<128x4x7x114x48xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<128x7x14x228x48xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<128x7x14x228x48xf32>) -> tensor<128x7x14x228x48xf32>
    %2 = bufferization.alloc_tensor() : tensor<1x1x1xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %4 = bufferization.alloc_tensor() : tensor<128x4x7x114x48xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<128x4x7x114x48xf32>) -> tensor<128x4x7x114x48xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.pooling_ndhwc_sum {dilations = dense<1> : tensor<3xi64>, strides = dense<2> : tensor<3xi64>, tag = "operation_3"} ins(%1, %3 : tensor<128x7x14x228x48xf32>, tensor<1x1x1xf32>) outs(%5 : tensor<128x4x7x114x48xf32>) -> tensor<128x4x7x114x48xf32>
    %8 = bufferization.alloc_tensor() : tensor<128x4x7x114x48xf32>
    %9 = bufferization.alloc_tensor() : tensor<128x4x7x114xf32>
    %10 = linalg.fill {tag = "operation_4"} ins(%cst : f32) outs(%9 : tensor<128x4x7x114xf32>) -> tensor<128x4x7x114xf32>
    %reduced = linalg.reduce ins(%7 : tensor<128x4x7x114x48xf32>) outs(%10 : tensor<128x4x7x114xf32>) dimensions = [4]  {tag = "operation_5"}
      (%in: f32, %init: f32) {
        %17 = arith.maximumf %in, %init : f32
        linalg.yield %17 : f32
      }
    %11 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"], library_call = "none"} ins(%7, %reduced : tensor<128x4x7x114x48xf32>, tensor<128x4x7x114xf32>) outs(%8 : tensor<128x4x7x114x48xf32>) attrs =  {tag = "operation_6"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %17 = arith.subf %in, %in_1 : f32
      %18 = math.exp %17 : f32
      linalg.yield %18 : f32
    } -> tensor<128x4x7x114x48xf32>
    %12 = bufferization.alloc_tensor() : tensor<1x1x1xf32>
    %13 = bufferization.alloc_tensor() : tensor<128x4x7x114x48xf32>
    %14 = linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>, tag = "operation_7"} ins(%11, %12 : tensor<128x4x7x114x48xf32>, tensor<1x1x1xf32>) outs(%13 : tensor<128x4x7x114x48xf32>) -> tensor<128x4x7x114x48xf32>
    %15 = call @nanoTime() : () -> i64
    %16 = arith.subi %15, %6 : i64
    call @printI64(%16) : (i64) -> ()
    call @printNewline() : () -> ()
    return %14 : tensor<128x4x7x114x48xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<128x4x7x114x48xf32>
    }
    return
  }
}
