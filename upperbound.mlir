#START_OPERATION
%1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<1024x1536xf32>) -> tensor<1024x1536xf32>
#START_NESTED_LOOPS
d0 0 1024 1 parallel
d1 0 1536 1 parallel
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
%3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<1024x56xf32>) -> tensor<1024x56xf32>
#START_NESTED_LOOPS
d0 0 1024 1 parallel
d1 0 56 1 parallel
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
%5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<1536x56xf32>) -> tensor<1536x56xf32>
#START_NESTED_LOOPS
d0 0 1536 1 parallel
d1 0 56 1 parallel
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
%7 = linalg.matmul_transpose_a {tag = "operation_3"} ins(%1, %3 : tensor<1024x1536xf32>, tensor<1024x56xf32>) outs(%5 : tensor<1536x56xf32>) -> tensor<1536x56xf32>
#START_NESTED_LOOPS
d0 0 1536 1 parallel
d1 0 56 1 parallel
d2 0 1024 1 reduction
#START_LOAD_DATA
d2, d0
d2, d1
d0, d1
#START_OP_COUNT
+ 1
- 0
* 1
/ 0
exp 0
#START_TAG
operation_3
#END_OPERATION






#START_OPERATION
%27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>], iterator_types = ["parallel", "reduction", "parallel"]} ins(%7, %extracted_slice : tensor<1536x56xf32>, tensor<?x56xf32>) outs(%extracted_slice_0 : tensor<1536x?xf32>) attrs =  {tag = "operation_4"} {
^bb0(%in: f32, %in_3: f32, %out: f32):
  %30 = arith.mulf %in, %in_3 : f32
  %31 = arith.addf %out, %30 : f32
  linalg.yield %31 : f32
} -> tensor<1536x?xf32>
#START_NESTED_LOOPS
d0 0 1536 1 parallel
d1 0 56 1 reduction
d2 0 -9223372036854775808 1 parallel
#START_LOAD_DATA
d0, d1
d2, d1
d0, d2
#START_OP_COUNT
+ 1
- 0
* 1
/ 0
exp 0
#START_TAG
operation_4
#END_OPERATION






#START_OPERATION
%28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>, affine_map<(d0, d1, d2) -> (d2, d0)>], iterator_types = ["parallel", "reduction", "parallel"]} ins(%27, %extracted_slice_1 : tensor<1536x?xf32>, tensor<?x8xf32>) outs(%extracted_slice_2 : tensor<1536x8xf32>) attrs =  {tag = "operation_5"} {
^bb0(%in: f32, %in_3: f32, %out: f32):
  %30 = arith.mulf %in, %in_3 : f32
  %31 = arith.addf %out, %30 : f32
  linalg.yield %31 : f32
} -> tensor<1536x8xf32>
#START_NESTED_LOOPS
d0 0 8 1 parallel
d1 0 -9223372036854775808 1 reduction
d2 0 1536 1 parallel
#START_LOAD_DATA
d2, d1
d1, d0
d2, d0
#START_OP_COUNT
+ 1
- 0
* 1
/ 0
exp 0
#START_TAG
operation_5
#END_OPERATION






#START_OPERATION
%23 = linalg.matmul_transpose_a {tag = "operation_6"} ins(%extracted_slice, %extracted_slice_0 : tensor<1536x64xf32>, tensor<1536x1xf32>) outs(%extracted_slice_1 : tensor<64x1xf32>) -> tensor<64x1xf32>
#START_NESTED_LOOPS
d0 0 64 1 parallel
d1 0 1 1 parallel
d2 0 1536 1 reduction
#START_LOAD_DATA
d2, d0
d2, d1
d0, d1
#START_OP_COUNT
+ 1
- 0
* 1
/ 0
exp 0
#START_TAG
operation_6
#END_OPERATION






#START_OPERATION
%21 = linalg.matmul_transpose_b {tag = "operation_7"} ins(%extracted_slice, %16 : tensor<1x28xf32>, tensor<112x28xf32>) outs(%extracted_slice_0 : tensor<1x112xf32>) -> tensor<1x112xf32>
#START_NESTED_LOOPS
d0 0 1 1 parallel
d1 0 112 1 parallel
d2 0 28 1 reduction
#START_LOAD_DATA
d0, d2
d1, d2
d0, d1
#START_OP_COUNT
+ 1
- 0
* 1
/ 0
exp 0
#START_TAG
operation_7
#END_OPERATION











#BEGIN_GRAPH
operation_0 --> operation_3
operation_1 --> operation_3
operation_2 --> operation_3
operation_3 --> operation_4
operation_4 --> operation_5
#END_GRAPH
########################################
#map = affine_map<(d0) -> (d0 * -16 + 150, 16)>
#map1 = affine_map<(d0) -> (d0 * 16)>
#map2 = affine_map<(d0) -> (d0 * 8)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map8 = affine_map<(d0) -> (d0 * 64)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<768x112xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<1024x1536xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<1024x1536xf32>) -> tensor<1024x1536xf32>
    %2 = bufferization.alloc_tensor() : tensor<1024x56xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<1024x56xf32>) -> tensor<1024x56xf32>
    %4 = bufferization.alloc_tensor() : tensor<1536x56xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<1536x56xf32>) -> tensor<1536x56xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.matmul_transpose_a {tag = "operation_3"} ins(%1, %3 : tensor<1024x1536xf32>, tensor<1024x56xf32>) outs(%5 : tensor<1536x56xf32>) -> tensor<1536x56xf32>
    %8 = bufferization.alloc_tensor() : tensor<150x56xf32>
    %9 = bufferization.alloc_tensor() : tensor<1536x150xf32>
    %10 = bufferization.alloc_tensor() : tensor<150x768xf32>
    %11 = bufferization.alloc_tensor() : tensor<1536x768xf32>
    %12 = scf.forall (%arg0, %arg1) in (96, 10) shared_outs(%arg2 = %11) -> (tensor<1536x768xf32>) {
      %21 = affine.min #map(%arg1)
      %22 = affine.apply #map1(%arg1)
      %23 = affine.apply #map2(%arg0)
      %24 = affine.apply #map2(%arg0)
      %25 = affine.apply #map1(%arg1)
      %26 = affine.apply #map1(%arg1)
      %extracted_slice = tensor.extract_slice %8[%25, 0] [%21, 56] [1, 1] : tensor<150x56xf32> to tensor<?x56xf32>
      %extracted_slice_0 = tensor.extract_slice %9[0, %26] [1536, %21] [1, 1] : tensor<1536x150xf32> to tensor<1536x?xf32>
      %27 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction", "parallel"]} ins(%7, %extracted_slice : tensor<1536x56xf32>, tensor<?x56xf32>) outs(%extracted_slice_0 : tensor<1536x?xf32>) attrs =  {tag = "operation_4"} {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %30 = arith.mulf %in, %in_3 : f32
        %31 = arith.addf %out, %30 : f32
        linalg.yield %31 : f32
      } -> tensor<1536x?xf32>
      %extracted_slice_1 = tensor.extract_slice %10[%22, %23] [%21, 8] [1, 1] : tensor<150x768xf32> to tensor<?x8xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[0, %24] [1536, 8] [1, 1] : tensor<1536x768xf32> to tensor<1536x8xf32>
      %28 = linalg.generic {indexing_maps = [#map4, #map6, #map7], iterator_types = ["parallel", "reduction", "parallel"]} ins(%27, %extracted_slice_1 : tensor<1536x?xf32>, tensor<?x8xf32>) outs(%extracted_slice_2 : tensor<1536x8xf32>) attrs =  {tag = "operation_5"} {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %30 = arith.mulf %in, %in_3 : f32
        %31 = arith.addf %out, %30 : f32
        linalg.yield %31 : f32
      } -> tensor<1536x8xf32>
      %29 = affine.apply #map2(%arg0)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %28 into %arg2[0, %29] [1536, 8] [1, 1] : tensor<1536x8xf32> into tensor<1536x768xf32>
      }
    }
    %13 = bufferization.alloc_tensor() : tensor<1536x28xf32>
    %14 = bufferization.alloc_tensor() : tensor<768x28xf32>
    %15 = scf.forall (%arg0, %arg1) in (12, 28) shared_outs(%arg2 = %14) -> (tensor<768x28xf32>) {
      %21 = affine.apply #map8(%arg0)
      %22 = affine.apply #map8(%arg0)
      %extracted_slice = tensor.extract_slice %12[0, %21] [1536, 64] [1, 1] : tensor<1536x768xf32> to tensor<1536x64xf32>
      %extracted_slice_0 = tensor.extract_slice %13[0, %arg1] [1536, 1] [1, 1] : tensor<1536x28xf32> to tensor<1536x1xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%22, %arg1] [64, 1] [1, 1] : tensor<768x28xf32> to tensor<64x1xf32>
      %23 = linalg.matmul_transpose_a {tag = "operation_6"} ins(%extracted_slice, %extracted_slice_0 : tensor<1536x64xf32>, tensor<1536x1xf32>) outs(%extracted_slice_1 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %24 = affine.apply #map8(%arg0)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %23 into %arg2[%24, %arg1] [64, 1] [1, 1] : tensor<64x1xf32> into tensor<768x28xf32>
      }
    }
    %16 = bufferization.alloc_tensor() : tensor<112x28xf32>
    %17 = bufferization.alloc_tensor() : tensor<768x112xf32>
    %18 = scf.forall (%arg0) in (768) shared_outs(%arg1 = %17) -> (tensor<768x112xf32>) {
      %extracted_slice = tensor.extract_slice %15[%arg0, 0] [1, 28] [1, 1] : tensor<768x28xf32> to tensor<1x28xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [1, 112] [1, 1] : tensor<768x112xf32> to tensor<1x112xf32>
      %21 = linalg.matmul_transpose_b {tag = "operation_7"} ins(%extracted_slice, %16 : tensor<1x28xf32>, tensor<112x28xf32>) outs(%extracted_slice_0 : tensor<1x112xf32>) -> tensor<1x112xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %21 into %arg1[%arg0, 0] [1, 112] [1, 1] : tensor<1x112xf32> into tensor<768x112xf32>
      }
    }
    %19 = call @nanoTime() : () -> i64
    %20 = arith.subi %19, %6 : i64
    call @printI64(%20) : (i64) -> ()
    call @printNewline() : () -> ()
    return %18 : tensor<768x112xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<768x112xf32>
    }
    return
  }
}
