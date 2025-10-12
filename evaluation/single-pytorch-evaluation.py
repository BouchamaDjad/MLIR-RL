import torch
import torch.jit
import time

"""Python File to measure execution times of nazim's dataset with pytorch optimizations
"""

# Function to measure execution time for an operation
def measure_time(func, num_trials=10):
    # Warm-up run
    for _ in range(10):
        func()
    
    # Measure time
    start = time.time()
    for _ in range(num_trials):
        func()
    
    end = time.time()
    
    return (end - start) / num_trials

# Define operations based on the image

def matmul_ops():
    operations = []
    matmul_configs = [
        (256, 2048, 1000),
        (256, 1280, 1000),
        (256, 1536, 1000),
        (256, 1408, 1000),
        (256, 768, 768),
        (256, 1024, 1024),
        (256, 768, 3072),
        (256, 256, 128),
        (256, 4096, 1024),
        (256, 1536, 4096),
        (256, 768, 2),
        (256, 2048, 2048),
        (256, 128, 256),
        (256, 256, 512),
        (256, 512, 1024)
    ]
    for _, (M, N, K) in enumerate(matmul_configs):
        A = torch.randn(M, N)
        B = torch.randn(N, K)
        op = lambda x=A, y=B: torch.matmul(x,y)
        
        @torch.jit.script
        def jit_linear(x, y):
            return torch.matmul(x,y)
        
        jit_op = lambda x=A, y=B: jit_linear(x,y)
        operations.append((f"Matmul ({M}-{N}-{K})", op, jit_op))

    return operations


def conv_ops():
    operations = []

    conv2d_configs = [
        ((256, 14, 14, 256), (3, 3, 256, 256),  1),
        ((256, 14, 14, 256), (1, 1, 256, 1024), 1),
        ((256, 28, 28, 128), (3, 3, 128, 128),  1),
        ((256, 28, 28, 128), (1, 1, 128, 512),  1),
        ((256, 28, 28, 512), (1, 1, 512, 128),  1),
        ((256, 14, 14, 128), (3, 3, 128, 32),   1),
        ((256, 7, 7,   128), (3, 3, 128, 32),   1),
        ((256, 16, 16, 256), (3, 3, 256, 256),  1),
        ((256, 14, 14, 576), (1, 1, 576, 576),  1),
        ((256, 28, 28, 128), (3, 3, 128, 32),   1),
        ((256, 14, 14, 336), (1, 1, 336, 336),  1),
        ((256, 56, 56, 64),  (3, 3, 64, 64),    1),
        ((256, 28, 28, 448), (1, 1, 448, 448),  1),
        ((256, 56, 56, 64),  (1, 1, 64, 256),   1),
        ((256, 128, 128, 16),(7, 7, 16, 8),     2),
        ((256, 64, 64, 64),  (3, 3, 64, 16),    1),
        ((256, 32, 32, 32),  (7, 7, 32, 256),   2),
        ((256, 230, 230, 3), (7, 7, 3, 64),     2),
    ]
    for _, ((N, H, W, C_in), (F, _, _, C_out), stride) in enumerate(conv2d_configs):
        x = torch.randn(N, C_in, H, W)
        conv = torch.nn.Conv2d(C_in, C_out, kernel_size=F, stride=stride, bias=False)

        @torch.jit.script
        def jit_linear(x):
            return conv(x)
        
        jit_op = lambda inp=x, layer=jit_linear: layer(inp)
        op = lambda inp=x, layer=conv: layer(inp)

        operations.append((f"Conv2d ({N}-{H}-{C_in}-{C_out}-{F})", op, jit_op))

    return operations


def pool_ops():
    operations = []

    maxpool_configs = [
        (256, 114, 114, 64),
        (256, 147, 147, 64),
        (256, 71, 71, 192),
        (256, 167, 167, 42),
        (256, 85, 85, 84),
        (256, 43, 43, 336),
        (256, 23, 23, 672),
        (256, 113, 113, 11),
        (256, 57, 57, 22),
        (256, 29, 29, 88),
    ]
    for _, (N, H, W, C) in enumerate(maxpool_configs):
        x = torch.randn(N, C, H, W)
        maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        @torch.jit.script
        def jit_linear(x):
            return maxpool(x)
        
        jit_op = lambda inp=x, layer=jit_linear: layer(inp)
        op = lambda inp=x, layer=maxpool: layer(inp)
        operations.append((f"Maxpool ({N}-{H}-{W}-{C})", op, jit_op))

    return operations



def add_ops():
    operations = []

    add_configs = [
        (256, 14, 14, 1024),
        (256, 28, 28, 512),
        (256, 7, 7, 2048),
        (256, 56, 56, 256),
        (256, 21, 21, 336),
        (256, 11, 11, 672),
        (256, 42, 42, 168),
        (256, 15, 15, 304),
        (256, 14, 14, 88),
        (256, 7, 7, 176),
    ]
    for _, (N, C, H, W) in enumerate(add_configs):
        A = torch.randn(N, C, H, W)
        B = torch.randn(N, C, H, W)
        
        @torch.jit.script
        def jit_linear(x, y):
            return x + y
        
        jit_op = lambda x=A, y=B: jit_linear(x,y)
        op = lambda x=A, y=B: x + y
        operations.append((f"Add ({N}-{H}-{W}-{C})", op,jit_op))

    return operations



def relu_ops():
    operations = []

    relu_configs = [
        (256, 2048),
        (256, 512),
        (256, 1000),
        (256, 100),
        (256, 10),
        (256, 57, 57, 64),
        (256, 74, 74, 64),
        (256, 36, 36, 192),
        (256, 85, 85, 42),
        (256, 43, 43, 84),
        (256, 23, 23, 336),
        (256, 14, 14, 672),
        (256, 29, 29, 22),
        (256, 14, 14, 88),
    ]
    for _, shape in enumerate(relu_configs):
        if len(shape) == 4:
            N, H, W, C = shape
            x = torch.randn(N,C,H,W)

            @torch.jit.script
            def jit_linear(x):
                return torch.relu(x)
            
            jit_op = lambda x=x: jit_linear(x)
            op = lambda inp=x: torch.relu(inp)
            operations.append((f"ReLU ({N}-{H}-{W}-{C})", op, jit_op))
        else:
            N,W = shape
            x = torch.randn(N,W)
            
            @torch.jit.script
            def jit_linear(x, y):
                return torch.matmul(x,y)
        
            jit_op = lambda x=x: jit_linear(x)
            op = lambda inp=x: torch.relu(inp)
            operations.append((f"ReLU ({N})", op, jit_op))

    return operations

def create_operations():
    for ops in [
        matmul_ops,
        pool_ops,
        add_ops,
        conv_ops,
        relu_ops
    ]:

        yield ops()

# Main script
def main():
    # Create operations
    print("Measuring execution times (in seconds)...")
    
    # Measure execution times for Base and JIT
    print(f"{'Operation':<40} {'Base':<15} {'JIT':<15}")

    with open("eval-pytorch-1.csv","w") as f:   
        f.write("Name,base-torch,torch-jit\n") 
        for operations in create_operations():
            print("-" * 70)

            for name, op, jit_op in operations:

                base_time = measure_time(op)

                jit_time = measure_time(jit_op)
            
                print(f"{name:<40} {base_time:<15.6f} {jit_time:<15.6f}")
                f.write(f"{name:<20},{base_time},{jit_time}\n")


if __name__ == "__main__":
    main()