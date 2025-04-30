#map = affine_map<(d0, d1) -> (d0 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @matmul() -> memref<256x5x14x64xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x7x56x256xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 7 {
        affine.for %arg2 = 0 to 56 {
          affine.for %arg3 = 0 to 256 {
            affine.store %cst, %alloc[%arg0, %arg1, %arg2, %arg3] : memref<256x7x56x256xf32>
          }
        }
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 3 {
        affine.store %cst, %alloc_0[%arg0, %arg1] : memref<3x3xf32>
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x5x54x256xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 5 {
        affine.for %arg2 = 0 to 54 {
          affine.for %arg3 = 0 to 256 {
            affine.store %cst, %alloc_1[%arg0, %arg1, %arg2, %arg3] : memref<256x5x54x256xf32>
          }
        }
      }
    }
    %0 = call @nanoTime() : () -> i64
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 5 {
        affine.for %arg2 = 0 to 54 {
          affine.for %arg3 = 0 to 256 {
            affine.for %arg4 = 0 to 3 {
              affine.for %arg5 = 0 to 3 {
                %3 = affine.apply #map(%arg1, %arg4)
                %4 = affine.apply #map(%arg2, %arg5)
                %5 = affine.load %alloc[%arg0, %3, %4, %arg3] : memref<256x7x56x256xf32>
                %6 = affine.load %alloc_1[%arg0, %arg1, %arg2, %arg3] : memref<256x5x54x256xf32>
                %7 = arith.addf %6, %5 : f32
                affine.store %7, %alloc_1[%arg0, %arg1, %arg2, %arg3] : memref<256x5x54x256xf32>
              }
            }
          }
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256x5x27x128xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 5 {
        affine.for %arg2 = 0 to 27 {
          affine.for %arg3 = 0 to 128 {
            affine.for %arg4 = 0 to 1 {
              affine.for %arg5 = 0 to 1 {
                %3 = affine.apply #map1(%arg2, %arg4)
                %4 = affine.apply #map1(%arg3, %arg5)
                %5 = affine.load %alloc_1[%arg0, %arg1, %3, %4] : memref<256x5x54x256xf32>
                %6 = affine.load %alloc_3[%arg0, %arg1, %arg2, %arg3] : memref<256x5x27x128xf32>
                %7 = arith.maximumf %6, %5 : f32
                affine.store %7, %alloc_3[%arg0, %arg1, %arg2, %arg3] : memref<256x5x27x128xf32>
              }
            }
          }
        }
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<256x5x14x64xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 5 {
        affine.for %arg2 = 0 to 14 {
          affine.for %arg3 = 0 to 64 {
            affine.for %arg4 = 0 to 1 {
              affine.for %arg5 = 0 to 1 {
                %3 = affine.apply #map1(%arg2, %arg4)
                %4 = affine.apply #map1(%arg3, %arg5)
                %5 = affine.load %alloc_3[%arg0, %arg1, %3, %4] : memref<256x5x27x128xf32>
                %6 = affine.load %alloc_5[%arg0, %arg1, %arg2, %arg3] : memref<256x5x14x64xf32>
                %7 = arith.maximumf %6, %5 : f32
                affine.store %7, %alloc_5[%arg0, %arg1, %arg2, %arg3] : memref<256x5x14x64xf32>
              }
            }
          }
        }
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<256x5x14x64xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 5 {
        affine.for %arg2 = 0 to 14 {
          affine.for %arg3 = 0 to 64 {
            affine.for %arg4 = 0 to 1 {
              affine.for %arg5 = 0 to 1 {
                %3 = affine.apply #map(%arg1, %arg4)
                %4 = affine.apply #map(%arg2, %arg5)
                %5 = affine.load %alloc_5[%arg0, %3, %4, %arg3] : memref<256x5x14x64xf32>
                %6 = affine.load %alloc_7[%arg0, %arg1, %arg2, %arg3] : memref<256x5x14x64xf32>
                %7 = arith.addf %6, %5 : f32
                affine.store %7, %alloc_7[%arg0, %arg1, %arg2, %arg3] : memref<256x5x14x64xf32>
              }
            }
          }
        }
      }
    }
    %1 = call @nanoTime() : () -> i64
    %2 = arith.subi %1, %0 : i64
    call @printI64(%2) : (i64) -> ()
    call @printNewline() : () -> ()
    memref.dealloc %alloc : memref<256x7x56x256xf32>
    memref.dealloc %alloc_0 : memref<3x3xf32>
    memref.dealloc %alloc_1 : memref<256x5x54x256xf32>
    memref.dealloc %alloc_2 : memref<1x1xf32>
    memref.dealloc %alloc_3 : memref<256x5x27x128xf32>
    memref.dealloc %alloc_4 : memref<1x1xf32>
    memref.dealloc %alloc_5 : memref<256x5x14x64xf32>
    memref.dealloc %alloc_6 : memref<1x1xf32>
    return %alloc_7 : memref<256x5x14x64xf32>
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