#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<3072x1536xf32> {
      %cst = arith.constant 1.000000e+00 : f32
      %cst_0 = arith.constant 2.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<3072x2048xf32>
      %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<3072x2048xf32>) -> tensor<3072x2048xf32>
      %2 = bufferization.alloc_tensor() : tensor<2048x1536xf32>
      %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<2048x1536xf32>) -> tensor<2048x1536xf32>
      %4 = bufferization.alloc_tensor() : tensor<3072x1536xf32>
      %5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<3072x1536xf32>) -> tensor<3072x1536xf32>
      %6 = call @nanoTime() : () -> i64
      %7 = bufferization.alloc_tensor() : tensor<3072x1536xf32>
      %8 = scf.forall (%arg0) in (1536) shared_outs(%arg1 = %7) -> (tensor<3072x1536xf32>) {
        %11 = affine.apply #map(%arg0)
        %12 = affine.apply #map(%arg0)
        %13 = affine.apply #map(%arg0)
        %extracted_slice = tensor.extract_slice %1[%12, 0] [2, 2048] [1, 1] : tensor<3072x2048xf32> to tensor<2x2048xf32>
        %extracted_slice_1 = tensor.extract_slice %5[%13, 0] [2, 1536] [1, 1] : tensor<3072x1536xf32> to tensor<2x1536xf32>
        %14 = linalg.matmul {tag = "operation_3"} ins(%extracted_slice, %3 : tensor<2x2048xf32>, tensor<2048x1536xf32>) outs(%extracted_slice_1 : tensor<2x1536xf32>) -> tensor<2x1536xf32>
        %extracted_slice_2 = tensor.extract_slice %arg1[%11, 0] [2, 1536] [1, 1] : tensor<3072x1536xf32> to tensor<2x1536xf32>
        %15 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<2x1536xf32>) outs(%extracted_slice_2 : tensor<2x1536xf32>) attrs =  {tag = "operation_4"} {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.negf %in : f32
          %18 = math.exp %17 : f32
          %19 = arith.addf %18, %cst : f32
          %20 = arith.divf %cst, %19 : f32
          linalg.yield %20 : f32
        } -> tensor<2x1536xf32>
        %16 = affine.apply #map(%arg0)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %15 into %arg1[%16, 0] [2, 1536] [1, 1] : tensor<2x1536xf32> into tensor<3072x1536xf32>
        }
      }
      %9 = call @nanoTime() : () -> i64
      %10 = arith.subi %9, %6 : i64
      call @printI64(%10) : (i64) -> ()
      call @printNewline() : () -> ()
      return %8 : tensor<3072x1536xf32>
    }
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @matmul() : () -> tensor<3072x1536xf32>
      }
      return
    }
  }
  
