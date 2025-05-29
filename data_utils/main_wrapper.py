import re
from rl_autoschedular.observation import __remove_duplicate_args

def read_file_stream(filename: str):
    """used for huge files"""
    with open(filename, 'r') as file:
        for line in file:
            yield line

def extract_element_shape(shape: str) -> str:
    matches = re.findall(r'tensor<(?:\d+x)*([fi]\d+)>',shape)
    if matches == []:
        raise ValueError("shape is wrong")
        
    return matches[0]

def nn_transform_wrapper(operation: str) -> str:
    """ adds a main functions that allocates tensors for the nn model's arguments

    Args:
        operations (str): the functions signature for the nn model's forward function

    Returns:
        str: the main function to be inserted
    
    """
        
    fields = re.findall(r"\s*\(([^())]+)\)\s*->\s*([^(]+)",  operation)[0]
        
    args, shapes = [], []
    for f in fields[0].split(', '):
        arg, shape = f.split(':')
        shapes.append(shape.strip())
        args.append(arg)

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

    args,  shapes = __remove_duplicate_args(args,  shapes)

    shapes.append(fields[1])

    #############################################################
    dims = []

    for shape in shapes:
    
        if shape.startswith("tensor"):
            arg_dims = list(map(int,  re.findall(r'\d+',  shape[7:-5])))
            dims.append( arg_dims )
    
        else:
            dims.append( -1 )

    #############################################################
    func_name = re.search(r"@(\w+)",operation).group(1)

    func_call = f"func.call @{func_name}({', '.join(args)}) : ({', '.join(shapes[:-1])}) -> {shapes[-1]}"

    #############################################################
    # All code:
    code = ''

    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printNewline()\n"
    code += "\n"
    code += "\n"
    code += "func.func @main(){\n"
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "\n"
    code += "    %val_f32 = arith.constant 2.00000e+00 : f32\n"
    code += "    %val_i64 = arith.constant 2 : i64\n"
    code += "    %zero = arith.constant 0.00000e+00 : f32\n"
    code += "\n"
    for arg, shape, arg_dims in zip(args, shapes, dims):
        # print_info(arg,shape,arg_dims)
        if arg_dims != -1:
            tmp_arg = f'%tmp_{arg[1:]}'
            code +=f"    {tmp_arg} = bufferization.alloc_tensor() : {shape}\n"
            element_type = extract_element_shape(shape)
            code +=f"    {arg} = linalg.fill ins(%val_{element_type} : {element_type}) outs({tmp_arg} : {shape}) -> {shape}\n"
        else:
            code +=f"    {arg} = arith.constant 2.00000e+00 : f32\n"
    
    code += "        \n"
    code += "    scf.for %i = %c0 to %n step %c1 {\n"
    code += "        %t0 = func.call @nanoTime() : () -> (i64)\n"
    code += "        \n"
    code += f"         %outputmain = {func_call}\n"
    code += "        \n"
    code += "        %t = func.call @nanoTime() : () -> (i64)\n"
    code += "        %delta = arith.subi %t, %t0 : i64\n"
    code += "        func.call @printI64(%delta) : (i64) -> ()\n"
    code += "        func.call @printNewline() : () -> ()\n"
    code += "        \n"
    code += "    }\n"
    code += "    return\n"
    code += "}\n"

    return code

def main_wrapper(filename,model_name, out):
    """Adds a main function to a model's code and writes the resulting code to the out file.

    The function reads the input file line by line, searches for the model function definition,
    and inserts a wrapper code after the function's return statement and closing brace.
    The wrapper code is generated using the `nn_transform_wrapper` function, which takes the
    function signature as input. The modified code is written to the specified output file.

    Args:
        filename (str): Path to the input file containing the model code.
        model_name (str): Name of the model function to be wrapped.
        out (str): Path to the output file where the modified code will be saved.
    """
    
    wrapped_code = ''
    return_found = False
    wrapped_code_added = False
    with open(out,"w") as o:
        for line in read_file_stream(filename):
            o.write(line)

            if not wrapped_code and model_name in line:
                signature = line[:-2].strip()

                wrapped_code = nn_transform_wrapper(signature)
                # print(wrapped_code)

            if not return_found and "return" in line:
                return_found = True

            if not wrapped_code_added and wrapped_code and return_found and "}" in line:
                o.write(wrapped_code)
                wrapped_code_added = True
                continue


if __name__ == "__main__":
    # filename = "../benchmarks/MobileNetV2_linalg_asm.mlir" 
    filename = "../benchmarks/resnet18_linalg_asm.mlir"
    # filename = "../benchmarks/VGG_linalg_asm.mlir"

    out = "../benchmarks/test.mlir"

    # model_name = "ResNet"
    # model_name = "MobileNetV2"
    # model_name = "VGG"

    main_wrapper(filename, "forward", out)

    