from random import randint, choice, shuffle, random
import re
import string

from rl_autoschedular.observation import __remove_duplicate_args


def choice_topped(choices, max_value):
    trials_left = 50
    n = choice(choices)
    while not (n <= max_value) and trials_left != 0:
        n = choice(choices)
        trials_left -= 1

    if trials_left == 0:
        return None
    return n

BATCH_SIZES = []
SIZES = []
HEIGHTS = []
CHANNELS = []
KERNELS = []
DILATIONS = []
STRIDES = []

def add(*args):
    # SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 3))])
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(4)])
    return f"linalg.add ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def add_nn(*args):
    if args:
        if len(args[0])==2:
            B,N = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:             
        B = choice(BATCH_SIZES)
        N = choice(HEIGHTS)



    operation = f"""
    linalg.generic {{indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel"]}} ins(%44, %10 : tensor<{B}x{N}xf32>, tensor<{N}xf32>) outs(%42 : tensor<{B}x{N}xf32>) {{
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %46 = arith.addf %in, %in_1 : f32
      linalg.yield %46 : f32
    }}
    """.strip()
    return operation


def sub(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.sub ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def max(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.max ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def mul(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.mul ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def abs(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.abs ins(%arg0: tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def ceil(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.ceil ins(%arg0 : tensor<{SHAPE}xf32>) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def copy_(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.copy ins(%arg0 : tensor<{SHAPE}xf32>) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def fill(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.fill ins(%arg0 : f32) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def transpose(*args):
    L = randint(1, 5)

    permutation = list(range(L))
    shuffle(permutation)

    SHAPE1 = [choice(HEIGHTS) for _ in range(L)]

    SHAPE2 = []
    for i in range(L):
        SHAPE2.append(SHAPE1[permutation[i]])

    SHAPE1 = "x".join(map(str, SHAPE1))
    SHAPE2 = "x".join(map(str, SHAPE2))

    return f"linalg.transpose ins(%input:tensor<{SHAPE1}xf32>) outs(%init:tensor<{SHAPE2}xf32>) permutation = {permutation}"


def batch_matmul(*args):
    if args:
        if len(args[0]) == 3:
            B,N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        B = choice(BATCH_SIZES)
        N = choice(HEIGHTS)
        K = choice(HEIGHTS)
    
    M = choice(HEIGHTS)

    return f"linalg.batch_matmul ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2 : tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_matmul_transpose_a(*args):
    if args:
        if len(args[0]) == 3:
            B,K,N = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        B = choice(BATCH_SIZES)
        N = choice(HEIGHTS)
        K = choice(HEIGHTS)
        
    M = choice(HEIGHTS)

    return f"linalg.batch_matmul_transpose_a ins(%arg0, %arg1: tensor<{B}x{K}x{N}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2: tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_matmul_transpose_b(*args):
    if args:
        if len(args[0]) == 3:
            B,N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        B = choice(BATCH_SIZES)
        N = choice(HEIGHTS)
        K = choice(HEIGHTS)

    M = choice(HEIGHTS)
    return f"linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{M}x{K}xf32>) outs(%arg2: tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_reduce_matmul(*args):
    if args:
        if len(args[0]) == 3:
            B,N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        B = choice(BATCH_SIZES)
        N = choice(HEIGHTS)
        K = choice(HEIGHTS)

    M = choice(HEIGHTS)
    return f"linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul(*args):
    if args:
        if len(args[0]) == 2:
            N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        N = choice(SIZES)
        K = choice(SIZES)
        
    M = choice(SIZES)

    return f"linalg.matmul ins(%arg0, %arg1 : tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%arg2 : tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul_transpose_a(*args):
    if args:
        if len(args[0]) == 2:
            K,N = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        N = choice(SIZES)
        K = choice(SIZES)

    M = choice(HEIGHTS)
    return f"linalg.matmul_transpose_a ins(%arg0, %arg1: tensor<{K}x{N}xf32>, tensor<{K}x{M}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul_transpose_b(*args):
    if args:
        if len(args[0]) == 2:
            N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        N = choice(SIZES)
        K = choice(SIZES)

    M = choice(HEIGHTS)
    return f"linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<{N}x{K}xf32>, tensor<{M}x{K}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def conv_1d(*args):
    if args:
        if len(args[0]) == 1:
            N = args[0][0]
        else:
            raise Exception("Skipped")
    else:
        N = choice(HEIGHTS)

    F = choice_topped(KERNELS, N)
    N_ = N - F + 1
    return f"linalg.conv_1d ins(%input, %filter : tensor<{N}xf32>, tensor<{F}xf32>) outs(%output : tensor<{N_}xf32>) -> tensor<{N_}xf32>"


def conv_1d_ncw_fcw(*args):
    # INPUT: NCW1
    # KERNL: FCW2
    # OUTPUT: (N, F, W1-W2+1)

    if args:
        if len(args[0]) == 3:
            N,C,W1 = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W1 = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    W2 = choice_topped(KERNELS, (W1 + 2 * padding - 1) // dilation - 1)

    W3 = ((W1 + 2 * padding - dilation * (W2 - 1) - 1) // stride) + 1

    return f"linalg.conv_1d_ncw_fcw {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W1}xf32>, tensor<{F}x{C}x{W2}xf32>) outs (%init: tensor<{N}x{F}x{W3}xf32>) -> tensor<{N}x{F}x{W3}xf32>"


def conv_1d_nwc_wcf(*args):
    # INPUT: NWC
    # KERNL: WCF
    # OUTPUT: (N, W1-W2+1, F)

    if args:
        if len(args[0]) == 3:
            N,W1,C = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W1 = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    W2 = choice_topped(KERNELS, (W1 + 2 * padding - 1) // dilation - 1)

    W3 = ((W1 + 2 * padding - dilation * (W2 - 1) - 1) // stride) + 1

    return f"linalg.conv_1d_nwc_wcf {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W1}x{C}xf32>, tensor<{W2}x{C}x{F}xf32>) outs (%init: tensor<{N}x{W3}x{F}xf32>) -> tensor<{N}x{W3}x{F}xf32>"


def conv_2d(*args):
    if args:
        if len(args[0]) == 2:
            H,W = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
    else:
        H, W = choice(HEIGHTS), choice(HEIGHTS)

    F1 = F2 = choice_topped(KERNELS, min(H - 2, W - 2))

    H_ = H - F1 + 1
    W_ = W - F2 + 1

    return f"linalg.conv_2d ins(%input, %filter: tensor<{H}x{W}xi32>, tensor<{F1}x{F2}xi32>) outs(%output: tensor<{H_}x{W_}xi32>) -> tensor<{H_}x{W_}xi32>"


def conv_2d_nchw_fchw(*args):
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,C,H,W = tuple(args[0])
        
        if H != W:            
            raise Exception("given shape is not accepted")
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)


    # W = choice(HEIGHTS)
    W = H

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_nchw_fchw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{F}x{C}x{KH}x{KW}xf32>) outs (%init: tensor<{N}x{F}x{H_}x{W_}xf32>) -> tensor<{N}x{F}x{H_}x{W_}xf32>"


def conv_2d_ngchw_fgchw(*args):
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,G,C,H,W = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        G = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_ngchw_fgchw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{G}x{C}x{H}x{W}xf32>, tensor<{G}x{F}x{C}x{KH}x{KW}xf32>) outs (%init: tensor<{N}x{G}x{F}x{H_}x{W_}xf32>) -> tensor<{N}x{G}x{F}x{H_}x{W_}xf32>"


def conv_2d_nhwc_fhwc(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_nhwc_fhwc {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{F}x{KH}x{KW}x{C}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{F}xf32>) -> tensor<{N}x{H_}x{W_}x{F}xf32>"


def conv_2d_nhwc_hwcf(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_nhwc_hwcf {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{KH}x{KW}x{C}x{F}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{F}xf32>) -> tensor<{N}x{H_}x{W_}x{F}xf32>"


def conv_3d(*args):
    if args:
        if len(args[0]) == 3:
            H,W,D = tuple(args[0])
        else:
            raise Exception("skipped")
    else:
        H, W, D = choice(HEIGHTS), choice(HEIGHTS), choice(HEIGHTS)

    F = choice_topped(KERNELS, min(H, W, D) - 2)

    H_ = H - F + 1
    W_ = W - F + 1
    D_ = D - F + 1

    return f"linalg.conv_3d ins(%input, %filter: tensor<{H}x{W}x{D}xf32>, tensor<{F}x{F}x{F}xf32>) outs(%output: tensor<{H_}x{W_}x{D_}xf32>) -> tensor<{H_}x{W_}x{D_}xf32>"


def conv_3d_ncdhw_fcdhw(*args):
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,C,H,W,D = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)
        D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = KD = choice_topped(
        KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1
    )

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (KD - 1) - 1) // stride) + 1

    return f"linalg.conv_3d_ncdhw_fcdhw {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}x{D}xf32>, tensor<{F}x{C}x{KH}x{KW}x{KD}xf32>) outs (%init: tensor<{N}x{F}x{H_}x{W_}x{D_}xf32>) -> tensor<{N}x{F}x{H_}x{W_}x{D_}xf32>"


def depthwise_conv_1d_ncw_cw(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,C,W= tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_ncw_cw {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{C}x{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def depthwise_conv_1d_nwc_wc(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,W,C= tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_nwc_wc {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}x{C}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"


def depthwise_conv_1d_nwc_wcm(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,W,C = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)
    
    M = choice(CHANNELS)
    

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_nwc_wcm {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{W_}x{C}x{M}xf32>"


def depthwise_conv_2d_nchw_chw(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,C,H,W = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nchw_chw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{C}x{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


def depthwise_conv_2d_nhwc_hwc(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nhwc_hwc {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{C}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def depthwise_conv_2d_nhwc_hwcm(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    M = choice(CHANNELS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nhwc_hwcm {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{H_}x{W_}x{C}x{M}xf32>"


def depthwise_conv_3d_ncdhw_cdhw(*args):
    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,C,D,H,W = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)
        D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ncdhw_cdhw {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{C}x{D}x{H}x{W}xf32>, tensor<{C}x{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{D_}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{D_}x{H_}x{W_}xf32>"


def depthwise_conv_3d_ndhwc_dhwc(*args):
    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,D,H,W,C = tuple(args[0])
            
    else:    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)
        D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ndhwc_dhwc {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}x{C}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def depthwise_conv_3d_ndhwc_dhwcm(*args):
    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,D,H,W,C = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)
        H = choice(HEIGHTS)
        D = choice(HEIGHTS)
    
    M = choice(CHANNELS)
    
    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ndhwc_dhwcm {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}x{M}xf32>"


def pooling_nchw_max(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,C,H,W = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nchw_max {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


def pooling_nchw_sum(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,C,H,W = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nchw_sum {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


def pooling_ncw_max(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,C,W = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ncw_max {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def pooling_ncw_sum(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,C,W = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ncw_sum {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def pooling_ndhwc_max(*args):
    if args:
        if len(args[0]) == 5:
            N,D,H,W,C = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        D = choice(HEIGHTS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W, D) - 1) // dilation - 1)

    D_ = (D - dilation * (K - 1) - 1) // stride + 1
    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ndhwc_max {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def pooling_ndhwc_min(*args):
    if args:
        if len(args[0]) == 5:
            N,D,H,W,C = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        D = choice(HEIGHTS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W, D) - 1) // dilation - 1)

    D_ = (D - dilation * (K - 1) - 1) // stride + 1
    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ndhwc_min {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def pooling_ndhwc_sum(*args):
    if args:
        if len(args[0]) == 5:
            N,D,H,W,C = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        D = choice(HEIGHTS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W, D) - 1) // dilation - 1)

    D_ = (D - dilation * (K - 1) - 1) // stride + 1
    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ndhwc_sum {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def pooling_nhwc_max(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nhwc_max {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def pooling_nhwc_min(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nhwc_min {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def pooling_nhwc_sum(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nhwc_sum {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def pooling_nwc_max(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,W,C = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nwc_max {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"


def pooling_nwc_sum(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,W,C = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nwc_sum {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"




def relu(*args):
    
    if random() < 0.25:
        if args:
            if len(args[0])==2:
                N,S = tuple(args[0])
            else:
                raise Exception("Skipped")
        else:
            N = choice(BATCH_SIZES)
            S = choice(CHANNELS)
        
        N = choice(BATCH_SIZES)
        S = choice(SIZES)
        SHAPE = f"{N}x{S}"
        
        relu_maps = """
        #map2 = affine_map<(d0, d1) -> (d0, d1)>
        """.strip()

        relu_operation = """
        linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%38 : tensor<SHAPExf32>) outs(%35 : tensor<SHAPExf32>) {
            ^bb0(%in: f32, %out: f32):
            %cst_1 = arith.constant 0.000000e+00 : f32
            %46 = arith.cmpf ugt, %in, %cst_1 : f32
            %47 = arith.select %46, %in, %cst_1 : f32
            linalg.yield %47 : f32
        } -> tensor<SHAPExf32>
        """.strip().replace('SHAPE', SHAPE)
        
    else:

        if args:
            if len(args[0])!=3:
                raise Exception("Skipped")
            
            N,C,W,W_ = tuple(args[0])

            if W != W_:
                raise Exception("Skipped")                

        else:
            N = choice(BATCH_SIZES)
            C = choice(CHANNELS)
            W = choice(HEIGHTS)
    
        
        SHAPE = f"{N}x{C}x{W}x{W}"
    
        relu_maps = """
        #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
        """.strip()

        relu_operation = """
        linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<SHAPExf32>) outs(%25 : tensor<SHAPExf32>) {
            ^bb0(%in: f32, %out: f32):
            %cst_1 = arith.constant 0.000000e+00 : f32
            %90 = arith.cmpf ugt, %in, %cst_1 : f32
            %91 = arith.select %90, %in, %cst_1 : f32
            linalg.yield %91 : f32
        } -> tensor<SHAPExf32>
        """.strip().replace('SHAPE', SHAPE)
        
    
    return relu_operation, relu_maps


def sigmoid(*args):
    # Always 2D tensor

    if args:
        if len(args[0])==2:
            N,S = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:
        N = choice(BATCH_SIZES)
        S = choice(SIZES)
    
    SHAPE = f"{N}x{S}"

    sigmoid_maps = """
    #map2 = affine_map<(d0, d1) -> (d0, d1)>
    """.strip()

    sigmoid_operation = """
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} 
    ins(%38 : tensor<SHAPExf32>) outs(%35 : tensor<SHAPExf32>) {
        ^bb0(%in: f32, %out: f32):
        %cst_1 = arith.constant 1.000000e+00 : f32
        %neg = arith.negf %in : f32
        %exp = math.exp %neg : f32
        %denom = arith.addf %cst_1, %exp : f32
        %sigmoid = arith.divf %cst_1, %denom : f32
        linalg.yield %sigmoid : f32
    } -> tensor<SHAPExf32>
    """.strip().replace('SHAPE', SHAPE)

    return sigmoid_operation, sigmoid_maps

def softmax(*args,dim=3):
    if not args:
        SIZE = [str(choice(SIZES)) for _ in range(dim)] 
    else:
        SIZE = list(map(str,args[0]))
        dim = len(SIZE)

    # Define the tensor shape
    SHAPE = f"{'x'.join(SIZE)}xf32"

    Fill_SHAPE = f"{'x'.join(SIZE[:-1])}xf32" if dim!=1 else "f32" 

    maps = f"""
    #map1 = affine_map<({",".join([f"d{i}" for i in range(dim)])}) -> ({",".join([f"d{i}" for i in range(dim)])})>
    #map2 = affine_map<({",".join([f"d{i}" for i in range(dim)])}) -> ({", ".join([f"d{i}" for i in range(dim - 1)])})>
"""

    additional_function = (f"""
    func.func private @softmax(%input: tensor<{SHAPE}>, %output: tensor<{SHAPE}>) -> tensor<{SHAPE}> attributes {{ "func.inline" = unit }} {{
    %zero = arith.constant 0.00000e+00 : f32
    // Allocate temporary tensors for max and sum computations\n"""
    f"""//%tmp_max = bufferization.alloc_tensor() : tensor<{Fill_SHAPE}>\n""" 
    f"""
    // Inline compute_max functionality
    // %filled = linalg.fill ins(%zero : f32) outs(%tmp_max : tensor<{Fill_SHAPE}>) -> tensor<{Fill_SHAPE}>
    %filled = bufferization.alloc_tensor() : tensor<{Fill_SHAPE}>
    %max = linalg.reduce ins(%input: tensor<{SHAPE}>)
                            outs(%filled: tensor<{Fill_SHAPE}>) 
                            dimensions = [{dim - 1}]
                            (%in: f32, %acc: f32) {{
      %max = arith.maximumf %in, %acc : f32
      linalg.yield %max : f32
    }}
    // Inline compute_exp_sum functionality\n"""
    f"""%result = linalg.generic {{
      indexing_maps = [#map1, #map2, #map1],
      iterator_types = [{", ".join(['"parallel"' for _ in range(dim)])}],
      library_call = "none"
    }} ins(%input, %max : tensor<{SHAPE}>, tensor<{Fill_SHAPE}>)
      outs(%output : tensor<{SHAPE}>) {{
      ^bb0(%in: f32, %max_t: f32, %out: f32):
        %diff = arith.subf %in, %max_t : f32
        %exp = math.exp %diff : f32
        linalg.yield %exp : f32
    }} -> tensor<{SHAPE}>

    return %result : tensor<{SHAPE}>
}}
""")
    return (
        f"""func.call @softmax(%arg0, %dst) : (tensor<{SHAPE}>, tensor<{SHAPE}>) -> tensor<{SHAPE}>"""
        ,(maps,additional_function)
    )

# TODO: clean the code and refactor it
def randomSubGraph(verbose=False):
    
    params = []
    shapes = []
    return_vars = []
    return_shapes = []
    core = ""


    total_maps = ""
    total_additional_function = ""

    iterations = list(range(5))
    iterations_end = 5
    for _ in iterations:        
        
        operation_name = choice(list(LINALG_OPERATION_GENERATORS.keys())) # TODO: Restriction on operators
        
        if verbose:
            print(f"\033[91m{operation_name=}\033[0m")

        if return_shapes and return_shapes[-1]:
            shape = list(map(int,return_shapes[-1][len("tensor<"):-1].split('x')[:-1]))
            if verbose:
                print(f'\033[33m{shape}\033[0m')
            
            try:
                res = LINALG_OPERATION_GENERATORS[operation_name](shape)
            except:
                if verbose:
                    print(f"\033[33mskipped\033[0m")
                if iterations_end > 10:
                    break
                
                iterations.append(iterations_end+1)
                iterations_end += 1
                
                continue
        else:
            res = LINALG_OPERATION_GENERATORS[operation_name]()
        
        if verbose:
            print(f"\033[92m{res}\033[0m")

        maps = ""
        additional_function = ""

        if isinstance(res, tuple):
            raw_operation, additional_tuple = res
            if isinstance(additional_tuple, tuple):
                maps, additional_function = additional_tuple
                
            else:
                maps = additional_tuple

        else:
            raw_operation = res

        # Handling maps with the same name from different generators
        maps_identifiers = re.findall(r"#(\w+)[^\w]",maps)
        for map_id in maps_identifiers:
            new_map = f"map{''.join([choice(string.digits) for _ in range(5)])}"
            
            
            maps = re.sub(rf'\b{map_id}\b', new_map, maps)
            additional_function = re.sub(rf"\b{map_id}\b",new_map, additional_function)
            raw_operation = re.sub(rf"\b{map_id}\b",new_map,raw_operation)

        # Handling additional functions with the same name (same generator called twice or user negligence)
        functions_identifiers = re.findall(r"@(\w+)[^\w]", additional_function)
        for func_id in functions_identifiers:
            new_func = f"{func_id}{''.join([choice(string.digits) for _ in range(5)])}"
            
            additional_function = re.sub(rf"\b{func_id}\b",new_func, additional_function)
            raw_operation = re.sub(rf"\b{func_id}\b",new_func,raw_operation)    


        total_maps += "\n" + maps
        total_additional_function += "\n" + additional_function

        args,args_shape = getShapes_Args(raw_operation)

        # change the input shape
        if return_vars != []:
            old_shape = args_shape[0]

            # if any([x in raw_operation for x in ["matmul", "conv"]]):
                # raw_operation = raw_operation.replace(old_shape, return_shapes[-1], 1)
                # args_shape[0] = return_shapes[-1]

            if all([x not in raw_operation for x in ["generic", "func.call"]]) and \
                not any([x in raw_operation for x in ["matmul", "conv","pool"]]):
                if verbose:
                    print("\033[91m general shape change executed \033[0m")

                raw_operation = raw_operation.replace(old_shape, return_shapes[-1])
                args_shape = [return_shapes[-1] for _ in range(len(args_shape))]


        # dealing with arguments with the same name from different generators
        new_args = []
        for i,arg in enumerate(args):
            if i == 0 and return_vars != []:
                new_arg = return_vars[-1]
                args_shape.pop(0)

            else:
                new_arg = f"{arg}{''.join([choice(string.digits) for _ in range(5)])}"
                new_args.append(new_arg)

            raw_operation = raw_operation.replace(arg,new_arg)

        
        if params == []:
            params.extend(new_args)
            shapes.extend(args_shape)

        else:
            for arg,shape in zip(new_args,args_shape):
                if "tensor" in shape:
                    core += f"{arg} = bufferization.alloc_tensor() : {shape}\n"
                else:
                    core += f"{arg} = arith.constant 1.00000e+00 : f32\n"

        return_var = f"%var{''.join([choice(string.digits) for _ in range(5)])}" # TODO: prod-cons links
        return_vars.append(return_var)
        
        return_shape = args_shape[-1]
        
        if verbose:
            print(f"\033[90m {return_shape=}\033[0m")
        return_shapes.append(return_shape)

        core += f"""{return_var} = {raw_operation} \n"""

    core += f"""return {return_vars[-1]} : {return_shapes[-1]}\n"""

    total_additional_function += f"""\nfunc.func private @myFunction({", ".join([f"{p}:{s}" for p,s in zip(params,shapes)])}) -> {return_shapes[-1]} {{        
        {core}
    }}"""

    if verbose:
        print(f'\033[94m{total_additional_function=}\033[0m')

    final_operation = f"""func.call @myFunction({",".join(params)}) : ({",".join(shapes)}) -> {return_shapes[-1]}"""

    return final_operation,(total_maps,total_additional_function)

def getShapes_Args(operation):
    ins_outs_pattern = "(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    if fields == []:
	# # TODO: Add shape extraction so that allocation snippet could be replicated
        fields = re.findall("(?:\(([^(]+)\))(?:\s*\->\s*([^(]+))", operation)[0]
        
        args,shapes = [],[]
        for f in fields[0].split(", "):
            shapes.append(f)
        # shapes.append(fields[1])

        args = re.findall("(?:@\w+\(([^)]+))",operation)[0].split(',')

        args = [arg.strip() for arg in args]
        shapes = [shape.strip() for shape in shapes]

    else:
        args, shapes = [], []
        for field in fields:
            args_field, shapes_field = field.split(':')
            args   += args_field.split(',')
            shapes += shapes_field.split(',')

        args = [arg.strip() for arg in args]
        shapes = [shape.strip() for shape in shapes]

        args, shapes = __remove_duplicate_args(args, shapes)

    return args,shapes


LINALG_OPERATION_GENERATORS = {
    "add": add,
    # "add_nn": add_nn,
    # "sub": sub,
    # "max": max,
    # "mul": mul,
    # "abs": abs,
    # "ceil": ceil,
    # "copy": copy_,
    # "fill": fill, # problem with the first arg being f32
    # "transpose": transpose,
    # "batch_matmul": batch_matmul,
    # "batch_matmul_transpose_a": batch_matmul_transpose_a,
    # "batch_matmul_transpose_b": batch_matmul_transpose_b,
    # "batch_reduce_matmul": batch_reduce_matmul,
    "matmul": matmul,
    # "matmul_transpose_a": matmul_transpose_a,
    # "matmul_transpose_b": matmul_transpose_b,
    # "conv_1d": conv_1d,
    # "conv_1d_ncw_fcw": conv_1d_ncw_fcw,j
    # "conv_1d_nwc_wcf": conv_1d_nwc_wcf,
    # "conv_2d": conv_2d, # Integer use
    "conv_2d_nchw_fchw": conv_2d_nchw_fchw,
    # "conv_2d_ngchw_fgchw": conv_2d_ngchw_fgchw,
    # "conv_2d_nhwc_fhwc": conv_2d_nhwc_fhwc,
    "conv_2d_nhwc_hwcf": conv_2d_nhwc_hwcf,
    # "conv_3d": conv_3d, to skip
    # "conv_3d_ncdhw_fcdhw": conv_3d_ncdhw_fcdhw,
    # "depthwise_conv_1d_ncw_cw": depthwise_conv_1d_ncw_cw,
    # "depthwise_conv_1d_nwc_wc": depthwise_conv_1d_nwc_wc,
    # "depthwise_conv_1d_nwc_wcm": depthwise_conv_1d_nwc_wcm,
    # "depthwise_conv_2d_nchw_chw": depthwise_conv_2d_nchw_chw,
    # "depthwise_conv_2d_nhwc_hwc": depthwise_conv_2d_nhwc_hwc,
    # "depthwise_conv_2d_nhwc_hwcm": depthwise_conv_2d_nhwc_hwcm,
    # "depthwise_conv_3d_ncdhw_cdhw": depthwise_conv_3d_ncdhw_cdhw,
    # "depthwise_conv_3d_ndhwc_dhwc": depthwise_conv_3d_ndhwc_dhwc,
    # "depthwise_conv_3d_ndhwc_dhwcm": depthwise_conv_3d_ndhwc_dhwcm,
    "pooling_nchw_max": pooling_nchw_max,
    "pooling_nchw_sum": pooling_nchw_sum,
    "pooling_ncw_max": pooling_ncw_max,
    "pooling_ncw_sum": pooling_ncw_sum,
    "pooling_ndhwc_max": pooling_ndhwc_max,
    "pooling_ndhwc_min": pooling_ndhwc_min,
    "pooling_ndhwc_sum": pooling_ndhwc_sum,
    "pooling_nhwc_max": pooling_nhwc_max,
    "pooling_nhwc_min": pooling_nhwc_min,
    "pooling_nhwc_sum": pooling_nhwc_sum,
    "pooling_nwc_max": pooling_nwc_max,
    "pooling_nwc_sum": pooling_nwc_sum,
    "relu": relu,
    # "softmax_1d": lambda: softmax(dim=1),
    "softmax_2d": lambda *args: softmax(*args,dim=2),
    # "softmax_3d": lambda *args: softmax(*args, dim=3),
    # "softmax_4d": lambda *args: softmax(*args, dim=4),
    "sigmoid": sigmoid
}