#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<128x108x48xf32> {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 2.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<128x120x96xf32>
      %1 = linalg.fill {tag = "operation_0"} ins(%cst_0 : f32) outs(%0 : tensor<128x120x96xf32>) -> tensor<128x120x96xf32>
      %2 = bufferization.alloc_tensor() : tensor<7xf32>
      %3 = linalg.fill {tag = "operation_1"} ins(%cst_0 : f32) outs(%2 : tensor<7xf32>) -> tensor<7xf32>
      %4 = bufferization.alloc_tensor() : tensor<128x114x96xf32>
      %5 = linalg.fill {tag = "operation_2"} ins(%cst_0 : f32) outs(%4 : tensor<128x114x96xf32>) -> tensor<128x114x96xf32>
      %6 = call @nanoTime() : () -> i64
      %7 = bufferization.alloc_tensor() : tensor<128x114x96xf32>
      %8 = bufferization.alloc_tensor() : tensor<128x114x96xf32>
      %9 = scf.forall (%arg0, %arg1) in (64, 57) shared_outs(%arg2 = %8) -> (tensor<128x114x96xf32>) {
        %22 = affine.apply #map(%arg0)
        %23 = affine.apply #map(%arg1)
        %24 = affine.apply #map(%arg0)
        %25 = affine.apply #map(%arg1)
        %26 = affine.apply #map(%arg0)
        %27 = affine.apply #map(%arg1)
        %extracted_slice = tensor.extract_slice %1[%24, %25, 0] [2, 8, 96] [1, 1, 1] : tensor<128x120x96xf32> to tensor<2x8x96xf32>
        %extracted_slice_1 = tensor.extract_slice %5[%26, %27, 0] [2, 2, 96] [1, 1, 1] : tensor<128x114x96xf32> to tensor<2x2x96xf32>
        %28 = linalg.pooling_nwc_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>, tag = "operation_3"} ins(%extracted_slice, %3 : tensor<2x8x96xf32>, tensor<7xf32>) outs(%extracted_slice_1 : tensor<2x2x96xf32>) -> tensor<2x2x96xf32>
        %extracted_slice_2 = tensor.extract_slice %arg2[%22, %23, 0] [2, 2, 96] [1, 1, 1] : tensor<128x114x96xf32> to tensor<2x2x96xf32>
        %29 = vector.transfer_read %28[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : tensor<2x2x96xf32>, vector<2x2x96xf32>
        %30 = affine.apply #map(%arg0)
        %31 = affine.apply #map(%arg1)
        %32 = vector.transfer_read %7[%30, %31, %c0], %cst {in_bounds = [true, true, true]} : tensor<128x114x96xf32>, vector<2x2x96xf32>
        %33 = arith.addf %29, %32 : vector<2x2x96xf32>
        %34 = vector.transfer_write %33, %extracted_slice_2[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<2x2x96xf32>, tensor<2x2x96xf32>
        %35 = affine.apply #map(%arg0)
        %36 = affine.apply #map(%arg1)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %34 into %arg2[%35, %36, 0] [2, 2, 96] [1, 1, 1] : tensor<2x2x96xf32> into tensor<128x114x96xf32>
        }
      }
      %10 = bufferization.alloc_tensor() : tensor<1xf32>
      %11 = bufferization.alloc_tensor() : tensor<128x114x48xf32>
      %12 = linalg.pooling_ncw_max {dilations = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>, tag = "operation_5"} ins(%9, %10 : tensor<128x114x96xf32>, tensor<1xf32>) outs(%11 : tensor<128x114x48xf32>) -> tensor<128x114x48xf32>
      %13 = bufferization.alloc_tensor() : tensor<128x114x48xf32>
      %14 = bufferization.alloc_tensor() : tensor<128x114xf32>
      %15 = linalg.fill {tag = "operation_6"} ins(%cst : f32) outs(%14 : tensor<128x114xf32>) -> tensor<128x114xf32>
      %reduced = linalg.reduce ins(%12 : tensor<128x114x48xf32>) outs(%15 : tensor<128x114xf32>) dimensions = [2]  {tag = "operation_7"}
        (%in: f32, %init: f32) {
          %22 = arith.maximumf %in, %init : f32
          linalg.yield %22 : f32
        }
      %16 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"], library_call = "none"} ins(%12, %reduced : tensor<128x114x48xf32>, tensor<128x114xf32>) outs(%13 : tensor<128x114x48xf32>) attrs =  {tag = "operation_8"} {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %22 = arith.subf %in, %in_1 : f32
        %23 = math.exp %22 : f32
        linalg.yield %23 : f32
      } -> tensor<128x114x48xf32>
      %17 = bufferization.alloc_tensor() : tensor<7xf32>
      %18 = bufferization.alloc_tensor() : tensor<128x108x48xf32>
      %19 = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>, tag = "operation_9"} ins(%16, %17 : tensor<128x114x48xf32>, tensor<7xf32>) outs(%18 : tensor<128x108x48xf32>) -> tensor<128x108x48xf32>
      %20 = call @nanoTime() : () -> i64
      %21 = arith.subi %20, %6 : i64
      call @printI64(%21) : (i64) -> ()
      call @printNewline() : () -> ()
      return %19 : tensor<128x108x48xf32>
    }
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @matmul() : () -> tensor<128x108x48xf32>
      }
      return
    }
  }
  
