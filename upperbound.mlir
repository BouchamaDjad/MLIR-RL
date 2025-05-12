#START_OPERATION
%1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<128x120x240x14x32xf32>) -> tensor<128x120x240x14x32xf32>
#START_NESTED_LOOPS
d0 0 128 1 parallel
d1 0 120 1 parallel
d2 0 240 1 parallel
d3 0 14 1 parallel
d4 0 32 1 parallel
#START_LOAD_DATA
#START_OP_COUNT
+ 0
- 0
* 0
/ 0
exp 0
#START_TAG
operation_0
#END_OPERATION






#START_OPERATION
%3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
#START_NESTED_LOOPS
d0 0 3 1 parallel
d1 0 3 1 parallel
d2 0 3 1 parallel
#START_LOAD_DATA
#START_OP_COUNT
+ 0
- 0
* 0
/ 0
exp 0
#START_TAG
operation_1
#END_OPERATION






#START_OPERATION
%5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<128x59x119x6x32xf32>) -> tensor<128x59x119x6x32xf32>
#START_NESTED_LOOPS
d0 0 128 1 parallel
d1 0 59 1 parallel
d2 0 119 1 parallel
d3 0 6 1 parallel
d4 0 32 1 parallel
#START_LOAD_DATA
#START_OP_COUNT
+ 0
- 0
* 0
/ 0
exp 0
#START_TAG
operation_2
#END_OPERATION






#START_OPERATION
%7 = linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>, strides = dense<2> : tensor<3xi64>, tag = "operation_3"} ins(%1, %3 : tensor<128x120x240x14x32xf32>, tensor<3x3x3xf32>) outs(%5 : tensor<128x59x119x6x32xf32>) -> tensor<128x59x119x6x32xf32>
#START_NESTED_LOOPS
d0 0 128 1 parallel
d1 0 59 1 parallel
d2 0 119 1 parallel
d3 0 6 1 parallel
d4 0 32 1 parallel
d5 0 3 1 reduction
d6 0 3 1 reduction
d7 0 3 1 reduction
#START_LOAD_DATA
d0, d1 * 2 + d5, d2 * 2 + d6, d3 * 2 + d7, d4
d0, d1, d2, d3, d4
#START_OP_COUNT
+ 0
- 0
* 0
/ 0
exp 0
#START_TAG
operation_3
#END_OPERATION






#START_OPERATION
%28 = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>, tag = "operation_4"} ins(%extracted_slice_2, %8 : tensor<2x1x3x?x1xf32>, tensor<1x1x1xf32>) outs(%extracted_slice_3 : tensor<2x1x3x?x1xf32>) -> tensor<2x1x3x?x1xf32>
#START_NESTED_LOOPS
d0 0 2 1 parallel
d1 0 1 1 parallel
d2 0 3 1 parallel
d3 0 -9223372036854775808 1 parallel
d4 0 1 1 parallel
d5 0 1 1 reduction
d6 0 1 1 reduction
d7 0 1 1 reduction
#START_LOAD_DATA
d0, d1 + d5, d2 + d6, d3 + d7, d4
d0, d1, d2, d3, d4
#START_OP_COUNT
+ 0
- 0
* 0
/ 0
exp 0
#START_TAG
operation_4
#END_OPERATION






#START_OPERATION
%23 = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<2> : tensor<3xi64>, tag = "operation_5"} ins(%22, %10 : tensor<8x3x3x5x1xf32>, tensor<3x3x3xf32>) outs(%extracted_slice_1 : tensor<8x1x1x2x1xf32>) -> tensor<8x1x1x2x1xf32>
#START_NESTED_LOOPS
d0 0 8 1 parallel
d1 0 1 1 parallel
d2 0 1 1 parallel
d3 0 2 1 parallel
d4 0 1 1 parallel
d5 0 3 1 reduction
d6 0 3 1 reduction
d7 0 3 1 reduction
#START_LOAD_DATA
d0, d1 * 2 + d5, d2 * 2 + d6, d3 * 2 + d7, d4
d0, d1, d2, d3, d4
#START_OP_COUNT
+ 0
- 0
* 0
/ 0
exp 0
#START_TAG
operation_5
#END_OPERATION











#BEGIN_GRAPH
operation_0 --> operation_3
operation_1 --> operation_3
operation_2 --> operation_3
#END_GRAPH
########################################
#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0) -> (-d0 + 5, 2)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<128x29x59x2x32xf32> {
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<128x120x240x14x32xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<128x120x240x14x32xf32>) -> tensor<128x120x240x14x32xf32>
    %2 = bufferization.alloc_tensor() : tensor<3x3x3xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    %4 = bufferization.alloc_tensor() : tensor<128x59x119x6x32xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<128x59x119x6x32xf32>) -> tensor<128x59x119x6x32xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>, strides = dense<2> : tensor<3xi64>, tag = "operation_3"} ins(%1, %3 : tensor<128x120x240x14x32xf32>, tensor<3x3x3xf32>) outs(%5 : tensor<128x59x119x6x32xf32>) -> tensor<128x59x119x6x32xf32>
    %8 = bufferization.alloc_tensor() : tensor<1x1x1xf32>
    %9 = bufferization.alloc_tensor() : tensor<128x59x119x6x32xf32>
    %10 = bufferization.alloc_tensor() : tensor<3x3x3xf32>
    %11 = bufferization.alloc_tensor() : tensor<128x29x59x2x32xf32>
    %12 = scf.forall (%arg0, %arg1, %arg2, %arg3) in (16, 29, 59, 32) shared_outs(%arg4 = %11) -> (tensor<128x29x59x2x32xf32>) {
      %15 = affine.apply #map(%arg0)
      %16 = affine.apply #map(%arg0)
      %17 = affine.apply #map1(%arg1)
      %18 = affine.apply #map1(%arg2)
      %19 = affine.apply #map(%arg0)
      %20 = affine.apply #map1(%arg1)
      %21 = affine.apply #map1(%arg2)
      %extracted_slice = tensor.extract_slice %7[%16, %17, %18, 0, %arg3] [8, 3, 3, 5, 1] [1, 1, 1, 1, 1] : tensor<128x59x119x6x32xf32> to tensor<8x3x3x5x1xf32>
      %extracted_slice_0 = tensor.extract_slice %9[%19, %20, %21, 0, %arg3] [8, 3, 3, 5, 1] [1, 1, 1, 1, 1] : tensor<128x59x119x6x32xf32> to tensor<8x3x3x5x1xf32>
      %22 = scf.for %arg5 = %c0 to %c8 step %c2 iter_args(%arg6 = %extracted_slice_0) -> (tensor<8x3x3x5x1xf32>) {
        %25 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<8x3x3x5x1xf32>) {
          %26 = scf.for %arg9 = %c0 to %c5 step %c2 iter_args(%arg10 = %arg8) -> (tensor<8x3x3x5x1xf32>) {
            %27 = affine.min #map2(%arg9)
            %extracted_slice_2 = tensor.extract_slice %extracted_slice[%arg5, %arg7, 0, %arg9, 0] [2, 1, 3, %27, 1] [1, 1, 1, 1, 1] : tensor<8x3x3x5x1xf32> to tensor<2x1x3x?x1xf32>
            %extracted_slice_3 = tensor.extract_slice %arg10[%arg5, %arg7, 0, %arg9, 0] [2, 1, 3, %27, 1] [1, 1, 1, 1, 1] : tensor<8x3x3x5x1xf32> to tensor<2x1x3x?x1xf32>
            %28 = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>, tag = "operation_4"} ins(%extracted_slice_2, %8 : tensor<2x1x3x?x1xf32>, tensor<1x1x1xf32>) outs(%extracted_slice_3 : tensor<2x1x3x?x1xf32>) -> tensor<2x1x3x?x1xf32>
            %inserted_slice = tensor.insert_slice %28 into %arg10[%arg5, %arg7, 0, %arg9, 0] [2, 1, 3, %27, 1] [1, 1, 1, 1, 1] : tensor<2x1x3x?x1xf32> into tensor<8x3x3x5x1xf32>
            scf.yield %inserted_slice : tensor<8x3x3x5x1xf32>
          }
          scf.yield %26 : tensor<8x3x3x5x1xf32>
        }
        scf.yield %25 : tensor<8x3x3x5x1xf32>
      }
      %extracted_slice_1 = tensor.extract_slice %arg4[%15, %arg1, %arg2, 0, %arg3] [8, 1, 1, 2, 1] [1, 1, 1, 1, 1] : tensor<128x29x59x2x32xf32> to tensor<8x1x1x2x1xf32>
      %23 = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<2> : tensor<3xi64>, tag = "operation_5"} ins(%22, %10 : tensor<8x3x3x5x1xf32>, tensor<3x3x3xf32>) outs(%extracted_slice_1 : tensor<8x1x1x2x1xf32>) -> tensor<8x1x1x2x1xf32>
      %24 = affine.apply #map(%arg0)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %23 into %arg4[%24, %arg1, %arg2, 0, %arg3] [8, 1, 1, 2, 1] [1, 1, 1, 1, 1] : tensor<8x1x1x2x1xf32> into tensor<128x29x59x2x32xf32>
      }
    }
    %13 = call @nanoTime() : () -> i64
    %14 = arith.subi %13, %6 : i64
    call @printI64(%14) : (i64) -> ()
    call @printNewline() : () -> ()
    return %12 : tensor<128x29x59x2x32xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<128x29x59x2x32xf32>
    }
    return
  }
}
