#map = affine_map<(d0, d1) -> (d0 + d1)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @myFunction(%arg0: memref<256x56x130x384xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<7x7xf32, strided<[?, ?], offset: ?>>, %arg2: memref<256x50x124x384xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<256x48x122x384xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 50 {
        affine.for %arg5 = 0 to 124 {
          affine.for %arg6 = 0 to 384 {
            affine.for %arg7 = 0 to 7 {
              affine.for %arg8 = 0 to 7 {
                %0 = affine.apply #map(%arg4, %arg7)
                %1 = affine.apply #map(%arg5, %arg8)
                %2 = affine.load %arg0[%arg3, %0, %1, %arg6] : memref<256x56x130x384xf32, strided<[?, ?, ?, ?], offset: ?>>
                %3 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<256x50x124x384xf32, strided<[?, ?, ?, ?], offset: ?>>
                %4 = arith.addf %3, %2 : f32
                affine.store %4, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<256x50x124x384xf32, strided<[?, ?, ?, ?], offset: ?>>
              }
            }
          }
        }
      }
    }
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x50x124x384xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<256x50x124x384xf32>
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 50 {
        affine.for %arg5 = 0 to 124 {
          affine.for %arg6 = 0 to 384 {
            %0 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<256x50x124x384xf32, strided<[?, ?, ?, ?], offset: ?>>
            %1 = affine.load %alloc[%arg3, %arg4, %arg5, %arg6] : memref<256x50x124x384xf32>
            %2 = arith.addf %0, %1 : f32
            affine.store %2, %alloc_0[%arg3, %arg4, %arg5, %arg6] : memref<256x50x124x384xf32>
          }
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x50x124xf32>
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 50 {
        affine.for %arg5 = 0 to 124 {
          affine.store %cst, %alloc_1[%arg3, %arg4, %arg5] : memref<256x50x124xf32>
        }
      }
    }
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 50 {
        affine.for %arg5 = 0 to 124 {
          affine.for %arg6 = 0 to 384 {
            %0 = affine.load %alloc_0[%arg3, %arg4, %arg5, %arg6] : memref<256x50x124x384xf32>
            %1 = affine.load %alloc_1[%arg3, %arg4, %arg5] : memref<256x50x124xf32>
            %2 = arith.maximumf %0, %1 : f32
            affine.store %2, %alloc_1[%arg3, %arg4, %arg5] : memref<256x50x124xf32>
          }
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x50x124x384xf32>
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 50 {
        affine.for %arg5 = 0 to 124 {
          affine.for %arg6 = 0 to 384 {
            %0 = affine.load %alloc_0[%arg3, %arg4, %arg5, %arg6] : memref<256x50x124x384xf32>
            %1 = affine.load %alloc_1[%arg3, %arg4, %arg5] : memref<256x50x124xf32>
            %2 = arith.subf %0, %1 : f32
            %3 = math.exp %2 : f32
            affine.store %3, %alloc_2[%arg3, %arg4, %arg5, %arg6] : memref<256x50x124x384xf32>
          }
        }
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<256x48x122x384xf32>
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 48 {
        affine.for %arg5 = 0 to 122 {
          affine.for %arg6 = 0 to 384 {
            affine.for %arg7 = 0 to 3 {
              affine.for %arg8 = 0 to 3 {
                %0 = affine.apply #map(%arg4, %arg7)
                %1 = affine.apply #map(%arg5, %arg8)
                %2 = affine.load %alloc_2[%arg3, %0, %1, %arg6] : memref<256x50x124x384xf32>
                %3 = affine.load %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<256x48x122x384xf32>
                %4 = arith.minimumf %3, %2 : f32
                affine.store %4, %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<256x48x122x384xf32>
              }
            }
          }
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<256x48x122xf32>
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 48 {
        affine.for %arg5 = 0 to 122 {
          affine.store %cst, %alloc_5[%arg3, %arg4, %arg5] : memref<256x48x122xf32>
        }
      }
    }
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 48 {
        affine.for %arg5 = 0 to 122 {
          affine.for %arg6 = 0 to 384 {
            %0 = affine.load %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<256x48x122x384xf32>
            %1 = affine.load %alloc_5[%arg3, %arg4, %arg5] : memref<256x48x122xf32>
            %2 = arith.maximumf %0, %1 : f32
            affine.store %2, %alloc_5[%arg3, %arg4, %arg5] : memref<256x48x122xf32>
          }
        }
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<256x48x122x384xf32>
    affine.for %arg3 = 0 to 256 {
      affine.for %arg4 = 0 to 48 {
        affine.for %arg5 = 0 to 122 {
          affine.for %arg6 = 0 to 384 {
            %0 = affine.load %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<256x48x122x384xf32>
            %1 = affine.load %alloc_5[%arg3, %arg4, %arg5] : memref<256x48x122xf32>
            %2 = arith.subf %0, %1 : f32
            %3 = math.exp %2 : f32
            affine.store %3, %alloc_6[%arg3, %arg4, %arg5, %arg6] : memref<256x48x122x384xf32>
          }
        }
      }
    }
    memref.dealloc %alloc : memref<256x50x124x384xf32>
    memref.dealloc %alloc_0 : memref<256x50x124x384xf32>
    memref.dealloc %alloc_1 : memref<256x50x124xf32>
    memref.dealloc %alloc_2 : memref<256x50x124x384xf32>
    memref.dealloc %alloc_3 : memref<3x3xf32>
    memref.dealloc %alloc_4 : memref<256x48x122x384xf32>
    memref.dealloc %alloc_5 : memref<256x48x122xf32>
    return %alloc_6 : memref<256x48x122x384xf32>
  }
  func.func @matmul() -> memref<256x48x122x384xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x56x130x384xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 56 {
        affine.for %arg2 = 0 to 130 {
          affine.for %arg3 = 0 to 384 {
            affine.store %cst_0, %alloc[%arg0, %arg1, %arg2, %arg3] : memref<256x56x130x384xf32>
          }
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<7x7xf32>
    affine.for %arg0 = 0 to 7 {
      affine.for %arg1 = 0 to 7 {
        affine.store %cst_0, %alloc_1[%arg0, %arg1] : memref<7x7xf32>
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x50x124x384xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 50 {
        affine.for %arg2 = 0 to 124 {
          affine.for %arg3 = 0 to 384 {
            affine.store %cst_0, %alloc_2[%arg0, %arg1, %arg2, %arg3] : memref<256x50x124x384xf32>
          }
        }
      }
    }
    %0 = call @nanoTime() : () -> i64
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 50 {
        affine.for %arg2 = 0 to 124 {
          affine.for %arg3 = 0 to 384 {
            affine.for %arg4 = 0 to 7 {
              affine.for %arg5 = 0 to 7 {
                %3 = affine.apply #map(%arg1, %arg4)
                %4 = affine.apply #map(%arg2, %arg5)
                %5 = affine.load %alloc[%arg0, %3, %4, %arg3] : memref<256x56x130x384xf32>
                %6 = affine.load %alloc_2[%arg0, %arg1, %arg2, %arg3] : memref<256x50x124x384xf32>
                %7 = arith.addf %6, %5 : f32
                affine.store %7, %alloc_2[%arg0, %arg1, %arg2, %arg3] : memref<256x50x124x384xf32>
              }
            }
          }
        }
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256x50x124x384xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<256x50x124x384xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 50 {
        affine.for %arg2 = 0 to 124 {
          affine.for %arg3 = 0 to 384 {
            %3 = affine.load %alloc_2[%arg0, %arg1, %arg2, %arg3] : memref<256x50x124x384xf32>
            %4 = affine.load %alloc_3[%arg0, %arg1, %arg2, %arg3] : memref<256x50x124x384xf32>
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %alloc_4[%arg0, %arg1, %arg2, %arg3] : memref<256x50x124x384xf32>
          }
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<256x50x124xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 50 {
        affine.for %arg2 = 0 to 124 {
          affine.store %cst, %alloc_5[%arg0, %arg1, %arg2] : memref<256x50x124xf32>
        }
      }
    }
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 50 {
        affine.for %arg2 = 0 to 124 {
          affine.for %arg3 = 0 to 384 {
            %3 = affine.load %alloc_4[%arg0, %arg1, %arg2, %arg3] : memref<256x50x124x384xf32>
            %4 = affine.load %alloc_5[%arg0, %arg1, %arg2] : memref<256x50x124xf32>
            %5 = arith.maximumf %3, %4 : f32
            affine.store %5, %alloc_5[%arg0, %arg1, %arg2] : memref<256x50x124xf32>
          }
        }
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<256x50x124x384xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 50 {
        affine.for %arg2 = 0 to 124 {
          affine.for %arg3 = 0 to 384 {
            %3 = affine.load %alloc_4[%arg0, %arg1, %arg2, %arg3] : memref<256x50x124x384xf32>
            %4 = affine.load %alloc_5[%arg0, %arg1, %arg2] : memref<256x50x124xf32>
            %5 = arith.subf %3, %4 : f32
            %6 = math.exp %5 : f32
            affine.store %6, %alloc_6[%arg0, %arg1, %arg2, %arg3] : memref<256x50x124x384xf32>
          }
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<256x48x122x384xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 48 {
        affine.for %arg2 = 0 to 122 {
          affine.for %arg3 = 0 to 384 {
            affine.for %arg4 = 0 to 3 {
              affine.for %arg5 = 0 to 3 {
                %3 = affine.apply #map(%arg1, %arg4)
                %4 = affine.apply #map(%arg2, %arg5)
                %5 = affine.load %alloc_6[%arg0, %3, %4, %arg3] : memref<256x50x124x384xf32>
                %6 = affine.load %alloc_8[%arg0, %arg1, %arg2, %arg3] : memref<256x48x122x384xf32>
                %7 = arith.minimumf %6, %5 : f32
                affine.store %7, %alloc_8[%arg0, %arg1, %arg2, %arg3] : memref<256x48x122x384xf32>
              }
            }
          }
        }
      }
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<256x48x122xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 48 {
        affine.for %arg2 = 0 to 122 {
          affine.store %cst, %alloc_9[%arg0, %arg1, %arg2] : memref<256x48x122xf32>
        }
      }
    }
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 48 {
        affine.for %arg2 = 0 to 122 {
          affine.for %arg3 = 0 to 384 {
            %3 = affine.load %alloc_8[%arg0, %arg1, %arg2, %arg3] : memref<256x48x122x384xf32>
            %4 = affine.load %alloc_9[%arg0, %arg1, %arg2] : memref<256x48x122xf32>
            %5 = arith.maximumf %3, %4 : f32
            affine.store %5, %alloc_9[%arg0, %arg1, %arg2] : memref<256x48x122xf32>
          }
        }
      }
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<256x48x122x384xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 48 {
        affine.for %arg2 = 0 to 122 {
          affine.for %arg3 = 0 to 384 {
            %3 = affine.load %alloc_8[%arg0, %arg1, %arg2, %arg3] : memref<256x48x122x384xf32>
            %4 = affine.load %alloc_9[%arg0, %arg1, %arg2] : memref<256x48x122xf32>
            %5 = arith.subf %3, %4 : f32
            %6 = math.exp %5 : f32
            affine.store %6, %alloc_10[%arg0, %arg1, %arg2, %arg3] : memref<256x48x122x384xf32>
          }
        }
      }
    }
    %1 = call @nanoTime() : () -> i64
    %2 = arith.subi %1, %0 : i64
    call @printI64(%2) : (i64) -> ()
    call @printNewline() : () -> ()
    memref.dealloc %alloc : memref<256x56x130x384xf32>
    memref.dealloc %alloc_1 : memref<7x7xf32>
    memref.dealloc %alloc_2 : memref<256x50x124x384xf32>
    memref.dealloc %alloc_3 : memref<256x50x124x384xf32>
    memref.dealloc %alloc_4 : memref<256x50x124x384xf32>
    memref.dealloc %alloc_5 : memref<256x50x124xf32>
    memref.dealloc %alloc_6 : memref<256x50x124x384xf32>
    memref.dealloc %alloc_7 : memref<3x3xf32>
    memref.dealloc %alloc_8 : memref<256x48x122x384xf32>
    memref.dealloc %alloc_9 : memref<256x48x122xf32>
    return %alloc_10 : memref<256x48x122x384xf32>
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

