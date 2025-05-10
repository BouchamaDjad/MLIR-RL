#START_OPERATION
%1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<256x15x150x56x96xf32>) -> tensor<256x15x150x56x96xf32>
#START_NESTED_LOOPS
d0 0 256 1 parallel
d1 0 15 1 parallel
d2 0 150 1 parallel
d3 0 56 1 parallel
d4 0 96 1 parallel
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
%3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
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
%5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<256x7x74x27x96xf32>) -> tensor<256x7x74x27x96xf32>
#START_NESTED_LOOPS
d0 0 256 1 parallel
d1 0 7 1 parallel
d2 0 74 1 parallel
d3 0 27 1 parallel
d4 0 96 1 parallel
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
%7 = linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>, strides = dense<2> : tensor<3xi64>, tag = "operation_3"} ins(%1, %3 : tensor<256x15x150x56x96xf32>, tensor<3x3x3xf32>) outs(%5 : tensor<256x7x74x27x96xf32>) -> tensor<256x7x74x27x96xf32>
#START_NESTED_LOOPS
d0 0 256 1 parallel
d1 0 7 1 parallel
d2 0 74 1 parallel
d3 0 27 1 parallel
d4 0 96 1 parallel
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
%10 = linalg.fill {tag = "operation_4"} ins(%cst : f32) outs(%9 : tensor<256x7x74x27xf32>) -> tensor<256x7x74x27xf32>
#START_NESTED_LOOPS
d0 0 256 1 parallel
d1 0 7 1 parallel
d2 0 74 1 parallel
d3 0 27 1 parallel
#START_LOAD_DATA
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
%reduced = linalg.reduce ins(%7 : tensor<256x7x74x27x96xf32>) outs(%10 : tensor<256x7x74x27xf32>) dimensions = [4]  {tag = "operation_5"}
  (%in: f32, %init: f32) {
    %17 = arith.maximumf %in, %init : f32
    linalg.yield %17 : f32
  }
#START_NESTED_LOOPS
d0 0 256 1 parallel
d1 0 7 1 parallel
d2 0 74 1 parallel
d3 0 27 1 parallel
d4 0 96 1 reduction
#START_LOAD_DATA
d0, d1, d2, d3, d4
d0, d1, d2, d3
#START_OP_COUNT
+ 0
- 0
* 0
/ 0
exp 0
#START_TAG
operation_5
#END_OPERATION






#START_OPERATION
%23 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d2, d1, d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d1, d0, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d1, d0, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"], library_call = "none"} ins(%extracted_slice, %extracted_slice_1 : tensor<256x7x?x27x2xf32>, tensor<256x7x?x27xf32>) outs(%extracted_slice_2 : tensor<256x7x?x27x2xf32>) attrs =  {tag = "operation_6"} {
^bb0(%in: f32, %in_3: f32, %out: f32):
  %26 = arith.subf %in, %in_3 : f32
  %27 = math.exp %26 : f32
  linalg.yield %27 : f32
} -> tensor<256x7x?x27x2xf32>
#START_NESTED_LOOPS
d0 0 -9223372036854775808 1 parallel
d1 0 7 1 parallel
d2 0 256 1 parallel
d3 0 27 1 parallel
d4 0 2 1 parallel
#START_LOAD_DATA
d2, d1, d0, d3, d4
d2, d1, d0, d3
#START_OP_COUNT
+ 0
- 1
* 0
/ 0
exp 1
#START_TAG
operation_6
#END_OPERATION






#START_OPERATION
%24 = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<2> : tensor<3xi64>, tag = "operation_7"} ins(%extracted_slice, %12 : tensor<16x7x1x3x8xf32>, tensor<1x1x1xf32>) outs(%extracted_slice_1 : tensor<16x4x1x2x8xf32>) -> tensor<16x4x1x2x8xf32>
#START_NESTED_LOOPS
d0 0 16 1 parallel
d1 0 4 1 parallel
d2 0 1 1 parallel
d3 0 2 1 parallel
d4 0 8 1 parallel
d5 0 1 1 reduction
d6 0 1 1 reduction
d7 0 1 1 reduction
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
operation_7
#END_OPERATION











#BEGIN_GRAPH
operation_0 --> operation_3
operation_1 --> operation_3
operation_2 --> operation_3
operation_3 --> operation_5
operation_4 --> operation_5
#END_GRAPH
########################################
#map = affine_map<(d0) -> (d0 * -8 + 74, 8)>
#map1 = affine_map<(d0) -> (d0 * 8)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d1, d0, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d1, d0, d3)>
#map5 = affine_map<(d0) -> (d0 * 16)>
#map6 = affine_map<(d0) -> (d0 * 4)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<256x4x37x14x96xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<256x15x150x56x96xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<256x15x150x56x96xf32>) -> tensor<256x15x150x56x96xf32>
    %2 = bufferization.alloc_tensor() : tensor<3x3x3xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    %4 = bufferization.alloc_tensor() : tensor<256x7x74x27x96xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<256x7x74x27x96xf32>) -> tensor<256x7x74x27x96xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>, strides = dense<2> : tensor<3xi64>, tag = "operation_3"} ins(%1, %3 : tensor<256x15x150x56x96xf32>, tensor<3x3x3xf32>) outs(%5 : tensor<256x7x74x27x96xf32>) -> tensor<256x7x74x27x96xf32>
    %8 = bufferization.alloc_tensor() : tensor<256x7x74x27x96xf32>
    %9 = bufferization.alloc_tensor() : tensor<256x7x74x27xf32>
    %10 = linalg.fill {tag = "operation_4"} ins(%cst : f32) outs(%9 : tensor<256x7x74x27xf32>) -> tensor<256x7x74x27xf32>
    %reduced = linalg.reduce ins(%7 : tensor<256x7x74x27x96xf32>) outs(%10 : tensor<256x7x74x27xf32>) dimensions = [4]  {tag = "operation_5"}
      (%in: f32, %init: f32) {
        %17 = arith.maximumf %in, %init : f32
        linalg.yield %17 : f32
      }
    %11 = scf.forall (%arg0, %arg1) in (10, 48) shared_outs(%arg2 = %8) -> (tensor<256x7x74x27x96xf32>) {
      %17 = affine.min #map(%arg0)
      %18 = affine.apply #map1(%arg0)
      %19 = affine.apply #map2(%arg1)
      %20 = affine.apply #map1(%arg0)
      %21 = affine.apply #map1(%arg0)
      %22 = affine.apply #map2(%arg1)
      %extracted_slice = tensor.extract_slice %7[0, 0, %18, 0, %19] [256, 7, %17, 27, 2] [1, 1, 1, 1, 1] : tensor<256x7x74x27x96xf32> to tensor<256x7x?x27x2xf32>
      %extracted_slice_1 = tensor.extract_slice %reduced[0, 0, %20, 0] [256, 7, %17, 27] [1, 1, 1, 1] : tensor<256x7x74x27xf32> to tensor<256x7x?x27xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[0, 0, %21, 0, %22] [256, 7, %17, 27, 2] [1, 1, 1, 1, 1] : tensor<256x7x74x27x96xf32> to tensor<256x7x?x27x2xf32>
      %23 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"], library_call = "none"} ins(%extracted_slice, %extracted_slice_1 : tensor<256x7x?x27x2xf32>, tensor<256x7x?x27xf32>) outs(%extracted_slice_2 : tensor<256x7x?x27x2xf32>) attrs =  {tag = "operation_6"} {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %26 = arith.subf %in, %in_3 : f32
        %27 = math.exp %26 : f32
        linalg.yield %27 : f32
      } -> tensor<256x7x?x27x2xf32>
      %24 = affine.apply #map1(%arg0)
      %25 = affine.apply #map2(%arg1)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %23 into %arg2[0, 0, %24, 0, %25] [256, 7, %17, 27, 2] [1, 1, 1, 1, 1] : tensor<256x7x?x27x2xf32> into tensor<256x7x74x27x96xf32>
      }
    }
    %12 = bufferization.alloc_tensor() : tensor<1x1x1xf32>
    %13 = bufferization.alloc_tensor() : tensor<256x4x37x14x96xf32>
    %14 = scf.forall (%arg0, %arg1, %arg2, %arg3) in (16, 37, 7, 12) shared_outs(%arg4 = %13) -> (tensor<256x4x37x14x96xf32>) {
      %17 = affine.apply #map5(%arg0)
      %18 = affine.apply #map2(%arg1)
      %19 = affine.apply #map6(%arg2)
      %20 = affine.apply #map1(%arg3)
      %21 = affine.apply #map5(%arg0)
      %22 = affine.apply #map2(%arg2)
      %23 = affine.apply #map1(%arg3)
      %extracted_slice = tensor.extract_slice %11[%17, 0, %18, %19, %20] [16, 7, 1, 3, 8] [1, 1, 1, 1, 1] : tensor<256x7x74x27x96xf32> to tensor<16x7x1x3x8xf32>
      %extracted_slice_1 = tensor.extract_slice %arg4[%21, 0, %arg1, %22, %23] [16, 4, 1, 2, 8] [1, 1, 1, 1, 1] : tensor<256x4x37x14x96xf32> to tensor<16x4x1x2x8xf32>
      %24 = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<2> : tensor<3xi64>, tag = "operation_7"} ins(%extracted_slice, %12 : tensor<16x7x1x3x8xf32>, tensor<1x1x1xf32>) outs(%extracted_slice_1 : tensor<16x4x1x2x8xf32>) -> tensor<16x4x1x2x8xf32>
      %25 = affine.apply #map5(%arg0)
      %26 = affine.apply #map2(%arg2)
      %27 = affine.apply #map1(%arg3)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %24 into %arg4[%25, 0, %arg1, %26, %27] [16, 4, 1, 2, 8] [1, 1, 1, 1, 1] : tensor<16x4x1x2x8xf32> into tensor<256x4x37x14x96xf32>
      }
    }
    %15 = call @nanoTime() : () -> i64
    %16 = arith.subi %15, %6 : i64
    call @printI64(%16) : (i64) -> ()
    call @printNewline() : () -> ()
    return %14 : tensor<256x4x37x14x96xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<256x4x37x14x96xf32>
    }
    return
  }
}
