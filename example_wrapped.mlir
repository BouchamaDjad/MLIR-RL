module attributes {torch.debug_module_name = "Net"} {
func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printMemrefF32(tensor<*xf32>)





func.func private @myFunction(%input05221:tensor<256x7x56x256xf32>, %filter91566:tensor<3x3xf32>, %init58557:tensor<256x5x54x256xf32>) -> tensor<256x5x14x64xf32> {        
        %var66390 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins (%input05221, %filter91566: tensor<256x7x56x256xf32>, tensor<3x3xf32>) outs (%init58557: tensor<256x5x54x256xf32>) -> tensor<256x5x54x256xf32> 
%filter06118 = bufferization.alloc_tensor() : tensor<1x1xf32>
%init42616 = bufferization.alloc_tensor() : tensor<256x5x27x128xf32>
%var85352 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins (%var66390, %filter06118: tensor<256x5x54x256xf32>, tensor<1x1xf32>) outs (%init42616: tensor<256x5x27x128xf32>) -> tensor<256x5x27x128xf32> 
%filter55042 = bufferization.alloc_tensor() : tensor<1x1xf32>
%init04465 = bufferization.alloc_tensor() : tensor<256x5x14x64xf32>
%var51922 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins (%var85352, %filter55042: tensor<256x5x27x128xf32>, tensor<1x1xf32>) outs (%init04465: tensor<256x5x14x64xf32>) -> tensor<256x5x14x64xf32> 
%filter67861 = bufferization.alloc_tensor() : tensor<1x1xf32>
%init35615 = bufferization.alloc_tensor() : tensor<256x5x14x64xf32>
%var87219 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins (%var51922, %filter67861: tensor<256x5x14x64xf32>, tensor<1x1xf32>) outs (%init35615: tensor<256x5x14x64xf32>) -> tensor<256x5x14x64xf32> 
return %var87219 : tensor<256x5x14x64xf32>

    }


func.func @matmul() -> tensor<256x5x14x64xf32>{

%val = arith.constant 2.00000e+00 : f32
%zero = arith.constant 0.00000e+00 : f32

%tmp_input05221 = bufferization.alloc_tensor() : tensor<256x7x56x256xf32>
%input05221 = linalg.fill ins(%val : f32) outs(%tmp_input05221 : tensor<256x7x56x256xf32>) -> tensor<256x7x56x256xf32>
%tmp_filter91566 = bufferization.alloc_tensor() : tensor<3x3xf32>
%filter91566 = linalg.fill ins(%val : f32) outs(%tmp_filter91566 : tensor<3x3xf32>) -> tensor<3x3xf32>
%tmp_init58557 = bufferization.alloc_tensor() : tensor<256x5x54x256xf32>
%init58557 = linalg.fill ins(%val : f32) outs(%tmp_init58557 : tensor<256x5x54x256xf32>) -> tensor<256x5x54x256xf32>

%t0 = func.call @nanoTime() : () -> (i64)

%return_arg = func.call @myFunction(%input05221,%filter91566,%init58557) : (tensor<256x7x56x256xf32>,tensor<3x3xf32>,tensor<256x5x54x256xf32>) -> tensor<256x5x14x64xf32>
%t = func.call @nanoTime() : () -> (i64)
%delta = arith.subi %t, %t0 : i64
%fp = arith.uitofp %delta : i64 to f64
// func.call @printFlops(%fp) : (f64) -> ()
func.call @printI64(%delta) : (i64) -> ()
func.call @printNewline() : () -> ()

return %return_arg : tensor<256x5x14x64xf32> 
}

func.func @main(){
    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
    %outputmain = func.call @matmul() : () -> tensor<256x5x14x64xf32>
    }
    return
}
}