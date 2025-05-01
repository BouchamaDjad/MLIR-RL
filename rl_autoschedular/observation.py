import numpy as np
import re
from typing import Optional
import os
from copy import copy
import subprocess
from rl_autoschedular import config as cfg
from rl_autoschedular.state import OperationFeatures, NestedLoopFeatures, BenchmarkFeatures, LoopFeatures, LoopNode


# ================================================ Public functions ================================================


# ================================================ Tree vecteur functions ========================================
def extract_function(code: str, function_name: str="myFunction") -> str:
    if function_name not in code:
        return code

    lines = code.split("\n")
    stack = 0
    start = False
    start_index = 0
    end_index = 0

    for i, line in enumerate(lines):
        if function_name in line and not start:
            start_index = i
            start = True
            stack = line.count("{") - line.count("}")
        elif start:
            stack += line.count("{") - line.count("}")
            if stack == 0:
                end_index = i
                break

    return "\n".join(lines[start_index:end_index + 1])
            
def build_loops_tree(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    
    lines = file_content.split('\n') if file_content else []
    
    pre_nodes_lines = []
    for line in lines:
        if "affine.for" in line:
            break

        pre_nodes_lines.append(line)

    pre_nodes_lines = [line for line in pre_nodes_lines if "affine_map" in line]

    pre_nodes_maps: dict[str, str] = {}

    for line in pre_nodes_lines:
        if "affine_map" in line:
                map_name, map_function = line.strip().split(' = ')
                map_function = map_function.split(' -> ')[1][1:-2]
                pre_nodes_maps[map_name] = map_function

    # print(pre_nodes_maps)

    lines = extract_function(file_content).split("\n")
    trees = parse_affine_loops(lines)
    for tree in trees: 
        process_tree(tree, pre_nodes_maps)
    
    return trees

def build_loops_tree_using_lowering(content: str, tmp_path: str):  
    content = __lower_linalg_to_loops_2(content, tmp_path)
    
    if not content:
        return None

    with open(tmp_path,"w") as f:
        f.write(content)

    tree = build_loops_tree(tmp_path)

    return tree


def parse_affine_loops(lines) -> list[LoopNode]:
    stack = []
    root_nodes = []
    collecting = False

    for line in lines:
        stripped = line.strip()

        if "affine.for" in stripped:
            parts = stripped.split()
            try:
                _, arg, _, lower, _, upper, _ = parts
                var_name = arg  # e.g., %arg0
            except (ValueError, IndexError):
                var_name = "unknown"

            parent = stack[-1] if stack else None
            node = LoopNode(var_name,upper,lower, parent=parent)
            

            # Add node to parent or as root
            if parent:
                parent.children.append(node)
            else:
                root_nodes.append(node)

            stack.append(node)
            node.instructions.append(line)
            continue

        # Add line to current loop's instructions
        if stack:
            stack[-1].instructions.append(line)

        # Closing brace ends current loop scope
        if "}" in stripped and stack:
            stack.pop()

    return root_nodes
    

def process_tree(node, maps = None):
    # print(maps)
    loop_features = extract_op_features_from_affine_code_tree(node,maps)
    node.vector = build_op_features_vector(loop_features)
    for child in node.children:
        process_tree(child,maps)
    return node

def extract_op_features_from_affine_code_tree(node, maps = None):
    """Get operation features from the raw operation.

    Args:
        lines: the code for the for loop

    Returns:
        OperationFeatures: operation features contained in the raw operation
    """

    lines = node.instructions
    
    # Build op features
    nested_loops = []
    op_count = {'+': 0, '-': 0, '*': 0, '/': 0, 'exp': 0}
    load_data = []
    store_data = []

    maps: dict[str, str] = maps if maps is not None else {}
    args_of_loops: list[str] = []
    args_of_map: dict[str, str] = {}

    for line in lines:

        if "affine_map" in line:
            map_name, map_function = line.strip().split(' = ')
            map_function = map_function.split(' -> ')[1][1:-2]
            maps[map_name] = map_function

        elif "affine.apply" in line:
            new_op, _, _, *map_name__args = line.strip().split(' ')
            map_name__args = ' '.join(map_name__args)
            s = map_name__args.index('(')
            map_name, args = map_name__args[:s], map_name__args[s + 1:-1].split(', ')
            mapping_string = copy(maps[map_name])
            for i in range(len(args)):
                mapping_string = mapping_string.replace(f'd{i}', args[i])
            # print(new_op, map_name, args, maps[map_name], mapping_string)
            args_of_map[new_op] = mapping_string

        elif "affine.load" in line:
            # print(line.strip().split(' ')[:-2])
            new_op, _, _, *alloc = line.strip().split(' ')[:-2]
            alloc = ' '.join(alloc)
            args = alloc.split('[')[1][:-1].split(', ')

            for i in range(len(args)):
                if args[i] in args_of_map:
                    args[i] = args_of_map[args[i]]

            load_data.append(args)

        elif "arith.addf" in line:
            op_count['+'] += 1
        elif "arith.mulf" in line:
            op_count['*'] += 1
        elif "arith.subf" in line:
            op_count['-'] += 1
        elif "arith.divf" in line:
            op_count['/'] += 1
        elif "math.exp" in line:
            op_count['exp'] += 1
    
    parent_node = node.parent
    while parent_node is not None:
        nested_loops.append(
                NestedLoopFeatures(
                    arg=parent_node.var_name,
                    lower_bound=int(parent_node.lower),
                    upper_bound=int(parent_node.upper),
                    step=1,
                    iterator_type='parallel'
                )
            )
        parent_node = parent_node.parent
        
    return LoopFeatures(
        op_count=op_count,
        load_data=load_data,
        store_data=store_data,
        nested_loops=nested_loops
    )

# ==================================================== end of tree vecteur functions ===================================

def build_op_features_vector_old(op_features: OperationFeatures):
    """Build the feature vector from the operation features dataclass.

    Args:
        op_features (OperationFeatures): the operation features

    Returns:
        np.ndarray: the feature vector
    """

    indices = [nested_loop.arg for nested_loop in op_features.nested_loops]
    indices_dim = {arg: i for (i, arg) in enumerate(indices)}

    # Nested loop features: (upper/lower bounds, step)
    nested_loops = np.zeros((cfg.max_num_loops,))
    for i, nested_loop in enumerate(op_features.nested_loops):
        if i == cfg.max_num_loops:
            break
        nested_loops[i] = nested_loop.upper_bound

    # load access matrices:

    load_data = op_features.load_data

    load_access_matrices = np.zeros((cfg.max_num_stores_loads, cfg.max_num_load_store_dim, cfg.max_num_loops), dtype=np.int16)

    for load_i, load in enumerate(load_data):
        if load_i == cfg.max_num_stores_loads:
            break
        dimensions_terms = [__formula_str_to_list(term) for term in load]
        for m, dimension_term in enumerate(dimensions_terms):
            for index, factor in dimension_term:
                if index in indices_dim:
                    n = indices_dim[index]
                    load_access_matrices[load_i, m, n] = factor

    # load access matrices:
    store_data = op_features.store_data

    store_access_matrices = np.zeros((cfg.max_num_load_store_dim, cfg.max_num_loops), dtype=np.int16)

    dimensions_terms = [__formula_str_to_list(term) for term in store_data]
    for m, dimension_term in enumerate(dimensions_terms):
        for index, factor in dimension_term:
            n = indices_dim[index]
            store_access_matrices[m, n] = factor

    # Operations count:
    operations_count = np.array(list(op_features.op_count.values()))

    # Feature vector:
    nested_loops = nested_loops.reshape(-1)
    load_access_matrices = load_access_matrices.reshape(-1)
    store_access_matrices = store_access_matrices.reshape(-1)

    # print('   ', nested_loops.shape, load_access_matrices.shape, store_access_matrices.shape, operations_count.shape)
    feature_vector = np.concatenate((nested_loops, load_access_matrices, store_access_matrices, operations_count))

    return feature_vector

def build_op_features_vector(op_features: OperationFeatures):
    """Build the feature vector from the operation features dataclass.

    Args:
        op_features (OperationFeatures): the operation features

    Returns:
        np.ndarray: the feature vector
    """   
    indices_size = min(cfg.max_num_loops, len(op_features.nested_loops))
    
    indices_array = [nested_loop.arg for nested_loop in op_features.nested_loops[:indices_size]]

    indices_dim = {arg: i for (i, arg) in enumerate([nested_loop.arg for nested_loop in op_features.nested_loops])}

    indices = np.zeros(shape=(cfg.max_num_loops,))
    indices[:indices_size] = [indices_dim[x] for x in indices_array]

    # redefinition in order to filter out args for loop not within the limit
    indices_dim = {arg: i for (arg, i) in indices_dim.items() if arg in indices_array}

    # Nested loop features: (upper/lower bounds, step)
    upper_bounds = np.zeros((cfg.max_num_loops,))
    for i, nested_loop in enumerate(op_features.nested_loops):
        if i == cfg.max_num_loops:
            break
        upper_bounds[i] = nested_loop.upper_bound

    # load access matrices:

    load_data = op_features.load_data

    load_access_matrices = np.zeros((cfg.max_num_stores_loads, cfg.max_num_load_store_dim, cfg.max_num_loops), dtype=np.int16)

    for load_i, load in enumerate(load_data):
        if load_i == cfg.max_num_stores_loads:
            break
        dimensions_terms = [__formula_str_to_list(term) for term in load]
        for m, dimension_term in enumerate(dimensions_terms):
            for index, factor in dimension_term:
                if index in indices_dim:
                    n = indices_dim[index]
                    load_access_matrices[load_i, m, n] = factor

    # load access matrices:
    store_data = op_features.store_data

    store_access_matrices = np.zeros((cfg.max_num_load_store_dim, cfg.max_num_loops), dtype=np.int16)

    dimensions_terms = [__formula_str_to_list(term) for term in store_data]
    for m, dimension_term in enumerate(dimensions_terms):
        for index, factor in dimension_term:
            n = indices_dim[index]
            store_access_matrices[m, n] = factor

    # Operations count:
    operations_count = np.array(list(op_features.op_count.values()))

    # Feature vector:
    upper_bounds = upper_bounds.reshape(-1)
    load_access_matrices = load_access_matrices.reshape(-1)
    store_access_matrices = store_access_matrices.reshape(-1)

    computation_vector = np.concatenate(
        (
            indices,
            load_access_matrices,
            store_access_matrices,
            operations_count
        )
    )

    # print('   ', nested_loops.shape, load_access_matrices.shape, store_access_matrices.shape, operations_count.shape)
    feature_vector = np.concatenate((upper_bounds, computation_vector))

    return feature_vector


def extract_op_features_from_affine_code(raw_operation: str, tmp_file_path: str, maps: Optional[str] = None, additional_function: Optional[str] = None):
    """Get operation features from the raw operation.

    Args:
        raw_operation (str): the raw operation
        tmp_file_path (str): the temporary file path to write the operation to

    Returns:
        OperationFeatures: operation features contained in the raw operation
    """
    # Get code as affine loops
    wrapped_operation = __function_wrapper(raw_operation,maps,additional_function)
    # wrapped_operation = __inline(wrapped_operation, tmp_file_path)
    loops = __lower_linalg_to_loops(wrapped_operation, tmp_file_path)
    lines = loops.split('\n') if loops else []

    # Build op features
    nested_loops = []
    op_count = {'+': 0, '-': 0, '*': 0, '/': 0, 'exp': 0}
    load_data = []
    store_data = []

    maps: dict[str, str] = {}
    args_of_loops: list[str] = []
    args_of_map: dict[str, str] = {}

    for line in lines:

        if "affine_map" in line:
            map_name, map_function = line.strip().split(' = ')
            map_function = map_function.split(' -> ')[1][1:-2]
            maps[map_name] = map_function

        elif "affine.apply" in line:
            new_op, _, _, *map_name__args = line.strip().split(' ')
            map_name__args = ' '.join(map_name__args)
            s = map_name__args.index('(')
            map_name, args = map_name__args[:s], map_name__args[s + 1:-1].split(', ')
            mapping_string = copy(maps[map_name])
            for i in range(len(args)):
                mapping_string = mapping_string.replace(f'd{i}', args[i])
            # print(new_op, map_name, args, maps[map_name], mapping_string)
            args_of_map[new_op] = mapping_string

        elif "affine.for" in line:
            _, arg, _, lower, _, upper, _ = line.strip().split(' ')
            # print(arg, lower, upper)
            # TODO: handle iterator types better
            nested_loops.append(
                NestedLoopFeatures(
                    arg=arg,
                    lower_bound=int(lower),
                    upper_bound=int(upper),
                    step=1,
                    iterator_type='parallel'
                )
            )
            args_of_loops.append(arg)

        elif "affine.load" in line:
            # print(line.strip().split(' ')[:-2])
            new_op, _, _, *alloc = line.strip().split(' ')[:-2]
            alloc = ' '.join(alloc)
            args = alloc.split('[')[1][:-1].split(', ')

            for i in range(len(args)):
                if args[i] in args_of_map:
                    args[i] = args_of_map[args[i]]

            load_data.append(args)

        elif "arith.addf" in line:
            op_count['+'] += 1
        elif "arith.mulf" in line:
            op_count['*'] += 1
        elif "arith.subf" in line:
            op_count['-'] += 1
        elif "arith.divf" in line:
            op_count['/'] += 1
        elif "math.exp" in line:
            op_count['exp'] += 1

    return OperationFeatures(
        raw_operation=raw_operation,
        op_count=op_count,
        load_data=load_data,
        store_data=store_data,
        nested_loops=nested_loops,
        op_trees=[] # TODO:
    )


def extract_bench_features_from_code(bench_name: str, code: str, root_execution_time: int, execution_time: int):
    """Extract benchmark features from the given code.

    Args:
        bench_name (str): the benchmark name
        code (str): the code to extract features from
        root_execution_time (int): the root execution time
        execution_time (int): the execution time

    Returns:
        BenchmarkFeatures: the extracted benchmark features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} -',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return __extract_bench_features_from_ast_result(bench_name, raw_ast_info, root_execution_time, execution_time)


def extract_bench_features_from_file(bench_name: str, file_path: str, root_execution_time: int, execution_time: int):
    """Extract benchmark features from the code in the file.

    Args:
        bench_name (str): the benchmark name
        file_path (str): the file path
        root_execution_time (int): the root execution time
        execution_time (int): the execution time

    Returns:
        BenchmarkFeatures: the extracted benchmark features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} {file_path}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return __extract_bench_features_from_ast_result(bench_name, raw_ast_info, root_execution_time, execution_time)


# def get_raw_ast_info(code: str, tmp_file_path: str):
#     """Get the raw AST information from the code

#     Args:
#         code (str): the code to get the AST information from
#         tmp_file_path (str): the temporary file path to write the code to

#     Returns:
#         str: the raw AST information
#     """

#     with open(tmp_file_path, "w") as file:
#         file.write(code)

#     result = subprocess.run(
#         f'{os.getenv("AST_DUMPER_BIN_PATH")} {tmp_file_path}',
#         shell=True,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )

#     return result.stdout.decode('utf-8')


# def get_ast(raw_ast_info: str):
#     """Get the AST from the raw AST information

#     Args:
#         raw_ast_info (str): the raw AST information

#     Returns:
#         dict: containing for each consumer operation, the producers and the operation
#         str: new code with tags
#     """

#     info, new_code = raw_ast_info.split("########################################")
#     operations_lines, graph_lines = info.split('#BEGIN_GRAPH')

#     operations_blocks = operations_lines.split('#START_OPERATION')
#     operations_blocks = [block.strip() for block in operations_blocks if block]

#     ast = {}
#     for block in operations_blocks:
#         block_lines = block.split('\n')

#         operation_tag = block_lines[-2]
#         operation = '\n'.join(block_lines[:-3])
#         operation = operation.split("#START_NESTED_LOOPS")[0]

#         ast[operation_tag] = {
#             'producers': [],
#             'operation': operation
#         }

#     graph_lines = graph_lines.split('\n')
#     graph_lines = [line.split(' --> ') for line in graph_lines if ' --> ' in line]

#     for (producer, consumer) in graph_lines:
#         ast[consumer]['producers'].insert(0, producer)

#     return ast, new_code.strip()


# ================================================ Private functions ================================================


def __formula_str_to_list(formula: str):
    """
    Turns assignement formula to a list of (index, factor)
    Example:
        formula = "%x1 - %x2 + %x3 * 5 - %x5 * 3"
        return [('%x1', 1), ('%x2', -1), ('%x3', 5), ('%x5', -3)]

    Args:
        formula (str): the formula as a string input

    Returns:
        list: list of (index, factor) pairs
    """
    formula = formula + ' +'
    terms = formula.split(' ')

    running_factor = 1
    running_term = None

    save = []

    for term in terms:

        if term.startswith('%'):
            running_term = term
        elif term == '+':
            save.append((running_term, running_factor))
            running_factor = 1
        elif term == '-':
            save.append((running_term, running_factor))
            running_factor = -1
        elif term.isnumeric():
            running_factor *= int(term)

    if save[0][0] is None:
        save = save[1:]

    return save


def __remove_duplicate_args(args: list[str], shapes: list[str]):
    """Removes duplicate pairs from the list of paired arguments with shapes
    Args:
        args (list[str]): list of arguments
        shapes (list[str]): list of shapes

    Returns:
        list[str]: list of arguments without duplicates
        list[str]: list of shapes without duplicates
    """
    args_shapes = list(zip(args, shapes))
    seen = set()
    result = []
    for item in args_shapes:
        if item not in seen:
            seen.add(item)
            result.append(item)

    args = [x for (x, _) in result]
    shapes = [x for (_, x) in result]
    return args, shapes


def __function_wrapper(operation: str, maps: Optional[str] = None, additional_function:Optional[str] = None):
    """Wraps the operation line in a function in order to be able to lower into loops

    Args:
        operation (str): the operation line to be wrapped
        maps (Optional[str], optional): the affine maps. Defaults to None.

    Returns:
        str: the wrapped operation
    """
    ins_outs_pattern = r"(?:ins|outs)\s*\(([^())]+)\)"
    fields: list[str] = re.findall(ins_outs_pattern, operation)

    if fields == []:
        returned_string = f"{maps}\n" if maps else ""
        returned_string += f"{additional_function}\n" if additional_function else ""
        return returned_string

    args: list[str] = []
    shapes: list[str] = []
    for field in fields:
        args_field, shapes_field = field.split(':')
        args += args_field.split(',')
        shapes += shapes_field.split(',')

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

    out_shape = shapes[-1]

    args, shapes = __remove_duplicate_args(args, shapes)

    args_str = ', '.join([f'{arg}: {shape}' for (arg, shape) in zip(args, shapes)])

    if maps is None:
        wrapped_operation = (
            f"{additional_function}\n" if additional_function else ""
            f"func.func @func_call({args_str}) -> {out_shape} {{\n"
            f"  %ret = {operation}\n"
            f"  return %ret : {out_shape}\n"
            "}"
        )
    else:
        wrapped_operation = (
            f"{maps}\n"
            f"{additional_function}\n" if additional_function else ""
            f"func.func @func_call({args_str}) -> {out_shape} {{\n"
            f"  %ret = {operation}\n"
            f"  return %ret : {out_shape}\n"
            "}"
        )

    return wrapped_operation


def __lower_linalg_to_loops(mlir_code: str, tmp_file_path: str):
    """
    Lower Linalg dialect code to Affine dialect

    Args:
        mlir_code (str): the MLIR code to be lowered to Affine dialect
        tmp_file_path (str): the temporary file to write the MLIR code to

    Returns:
        Optional[str]: the lowered code with affine dialect
    """
    # Write the MLIR code to a temporary file
    with open(tmp_file_path, "w") as file:
        file.write(mlir_code)

    # Lower the Linalg dialect code to Affine dialect
    out = os.popen(f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims --one-shot-bufferize=bufferize-function-boundaries --finalizing-bufferize --buffer-deallocation-pipeline --convert-linalg-to-affine-loops {tmp_file_path}").read()

    if out != '':
        return out
    else:
        return None

def __lower_linalg_to_loops_2(mlir_code: str, tmp_file_path: str):
    """
    Lower Linalg dialect code to Affine dialect

    Args:
        mlir_code (str): the MLIR code to be lowered to Affine dialect
        tmp_file_path (str): the temporary file to write the MLIR code to

    Returns:
        Optional[str]: the lowered code with affine dialect
    """
    # Write the MLIR code to a temporary file
    with open(tmp_file_path, "w") as file:
        file.write(mlir_code)

    # Lower the Linalg dialect code to Affine dialect
    # removed --linalg-fuse-elementwise-ops
    out = os.popen(f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt --linalg-fold-unit-extent-dims --one-shot-bufferize=bufferize-function-boundaries --finalizing-bufferize --buffer-deallocation-pipeline --convert-linalg-to-affine-loops {tmp_file_path}").read()

    if out != '':
        return out
    else:
        return None


def __extract_bench_features_from_ast_result(bench_name: str, raw_ast_info: str, root_execution_time: int, execution_time: int):
    """Extracts benchmark features from the code's AST result and execution time.

    Args:
        bench_name (str): the benchmark name
        raw_ast_info (str): the raw AST information
        root_execution_time (int): the root execution time
        execution_time (int): the execution time

    Returns:
        BenchmarkFeatures: extracted benchmark features
    """
    info, full_code = raw_ast_info.split("########################################")
    # exec_time = lower_and_run_code(full_code)
    operations_lines, _ = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    ops_tags = []
    operations = {}
    for operation_block in operations_blocks:
        nested_loops = []
        op_count = {}
        load_data = []
        store_data = []

        operation, rest = operation_block.split("#START_NESTED_LOOPS")

        nested_loops_str, rest = rest.split("#START_LOAD_DATA")
        loop_args = []
        for nested_loop_str in nested_loops_str.strip().split("\n"):
            if not nested_loop_str:
                continue
            arg, low, high, step, iter = nested_loop_str.strip().split(" ")
            nested_loops.append(NestedLoopFeatures(
                arg=f'%{arg}',
                lower_bound=int(low),
                upper_bound=int(high),
                step=int(step),
                iterator_type=iter
            ))
            loop_args.append(arg)

        loads_data_str, rest = rest.split("#START_OP_COUNT")
        for loop_arg in loop_args:
            loads_data_str = loads_data_str.replace(loop_arg, f'%{loop_arg}')
        for load_data_str in loads_data_str.strip().split("\n"):
            if not load_data_str:
                continue
            load_data.append(load_data_str.split(", "))

        ops_count_str, rest = rest.split("#START_TAG")
        for op_count_str in ops_count_str.strip().split("\n"):
            op, count = op_count_str.strip().split(" ")
            op_count[op] = int(count)

        operation_tag = rest.strip().split("\n")[0]
        ops_tags.append(operation_tag)
        operations[operation_tag] = OperationFeatures(
            raw_operation=operation,
            op_count=op_count,
            load_data=load_data,
            store_data=store_data,
            nested_loops=nested_loops,
            op_trees=[]
        )

    return BenchmarkFeatures(
        bench_name=bench_name,
        code=full_code,
        operation_tags=ops_tags,
        operations=operations,
        root_exec_time=root_execution_time,
        exec_time=execution_time
    )

def transform_wrapper(operation, maps: Optional[str]=None, additional_function: Optional[str] = None):

    ins_outs_pattern = "(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    if fields == [] and additional_function is not None:
	# # TODO: Add shape extraction so that allocation snippet could be replicated
        fields = re.findall("(?:\(([^(]+)\))(?:\s*\->\s*([^(]+))", operation)[0]
        
        args,shapes = [],[]
        for f in fields[0].split(","):
            shapes.append(f.strip())
        shapes.append(fields[1])

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
    
    # print_info(args,shapes)
    
    #############################################################
    # consts:
    dims = []
    unique_dims = set()
    for shape in shapes:
        if shape.startswith("tensor"):
            arg_dims = list(map(int, re.findall(r'\d+', shape[7:-5])))
            dims.append( arg_dims )
            unique_dims = unique_dims.union(arg_dims)
        else: # shape == "f32"
            dims.append( -1 )
            unique_dims = unique_dims.union([-1])

    unique_dims = sorted(list(unique_dims))

    # print(unique_dims)
    
    consts_snippet = ""
    for dim in unique_dims:
        if dim != -1:
            consts_snippet += f"  %c{dim} = arith.constant {dim} : index\n"

    #############################################################
    # allocations:

    # allocations_snippet = ""

    # for arg, shape, arg_dims in zip(args, shapes, dims):
    #     # print(arg, shape, arg_dims)
    #     if shape.startswith("tensor"):
    #         n = shape.count("x")
    #         temp_shape = "tensor<" + "?x"*n + shape[-4:] # f32> or i64> ir i32>
    #         alloc_params = ", ".join([f"%c{dim}" for dim in arg_dims])
    #         allocations_snippet += f"  {arg}_temp = bufferization.alloc_tensor({alloc_params}) : {temp_shape}\n"
    #         allocations_snippet += f"  {arg} = tensor.cast {arg}_temp : {temp_shape} to {shape}\n"
    #     else:
    #         # print(arg, shape, arg_dims)
    #         allocations_snippet += f"  {arg} = arith.constant 1.00000e+00 : f32\n"

    # print(allocations_snippet)

    #############################################################
    # function call:

    # function_call_snippet = f"  %ret_arg = func.call @func_call({', '.join(args)}) : ({', '.join(shapes)}) -> ({shapes[-1]})"

    #############################################################
    # All code:

    code = ""
    if maps is not None:
        code += f"{maps}\n"
    code += 'module attributes {torch.debug_module_name = "Net"} {\n'
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printFlops(f64)\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printNewline()\n"
    code += "func.func private @printMemrefF32(tensor<*xf32>)\n"
    code += f"{additional_function}\n" if additional_function else ""
    code += "\n"
    code += "\n"
    code +=f"func.func @matmul() -> {shapes[-1]}{{\n"
    code += "\n"
    code += "%val = arith.constant 2.00000e+00 : f32\n"
    code += "%zero = arith.constant 0.00000e+00 : f32\n"
    code += "\n"
    
    # code +=f"%out = bufferization.alloc_tensor() : tensor<{N}x{K}xf32>\n"
    # code +=f"%A = linalg.fill ins(%val : f32) outs(%out : tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32>\n"
    for arg, shape, arg_dims in zip(args, shapes, dims):
        # print_info(arg,shape,arg_dims)
        if shape != 'f32':
            tmp_arg = f'%tmp_{arg[1:]}'
            code +=f"{tmp_arg} = bufferization.alloc_tensor() : {shape}\n"
            code +=f"{arg} = linalg.fill ins(%val : f32) outs({tmp_arg} : {shape}) -> {shape}\n"
        else:
            code +=f"{arg} = arith.constant 2.00000e+00 : f32\n"
    
    code += "\n"
    code += "%t0 = func.call @nanoTime() : () -> (i64)\n"
    code += "\n"
    
    # code +=f"%D = linalg.matmul ins(%A, %B: tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%C: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>\n"
    code += f"%return_arg = {operation}"
    
    code += "\n"
    code += "%t = func.call @nanoTime() : () -> (i64)\n"
    code += "%delta = arith.subi %t, %t0 : i64\n"
    code += "%fp = arith.uitofp %delta : i64 to f64\n"
    code += "// func.call @printFlops(%fp) : (f64) -> ()\n"
    code += "func.call @printI64(%delta) : (i64) -> ()\n"
    code += "func.call @printNewline() : () -> ()\n"
    code += "\n"
    code +=f"return %return_arg : {shapes[-1]} \n"
    code += "}\n"
    code += "\n"
    code += "func.func @main(){\n"
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "    scf.for %i = %c0 to %n step %c1 {\n"
    code +=f"    %outputmain = func.call @matmul() : () -> {shapes[-1]}\n"
    code += "    }\n"
    code += "    return\n"
    code += "}\n"
    code += "}\n"

    return code

def transform_wrapper_2(operation:str, full_code:str, additional_function: Optional[str] = None):

    args, shapes = __getArgsShapes(operation, additional_function)
    
    # print_info(args,shapes)
    
    #############################################################
    # consts:
    dims = []
    unique_dims = set()
    for shape in shapes:
        if shape.startswith("tensor"):
            arg_dims = list(map(int, re.findall(r'\d+', shape[7:-5])))
            dims.append( arg_dims )
            unique_dims = unique_dims.union(arg_dims)
        else: # shape == "f32"
            dims.append( -1 )
            unique_dims = unique_dims.union([-1])

    unique_dims = sorted(list(unique_dims))

    # print(unique_dims)
    
    consts_snippet = ""
    for dim in unique_dims:
        if dim != -1:
            consts_snippet += f"  %c{dim} = arith.constant {dim} : index\n"

    #############################################################
    # allocations:

    # allocations_snippet = ""

    # for arg, shape, arg_dims in zip(args, shapes, dims):
    #     # print(arg, shape, arg_dims)
    #     if shape.startswith("tensor"):
    #         n = shape.count("x")
    #         temp_shape = "tensor<" + "?x"*n + shape[-4:] # f32> or i64> ir i32>
    #         alloc_params = ", ".join([f"%c{dim}" for dim in arg_dims])
    #         allocations_snippet += f"  {arg}_temp = bufferization.alloc_tensor({alloc_params}) : {temp_shape}\n"
    #         allocations_snippet += f"  {arg} = tensor.cast {arg}_temp : {temp_shape} to {shape}\n"
    #     else:
    #         # print(arg, shape, arg_dims)
    #         allocations_snippet += f"  {arg} = arith.constant 1.00000e+00 : f32\n"

    # print(allocations_snippet)

    #############################################################
    # function call:

    # function_call_snippet = f"  %ret_arg = func.call @func_call({', '.join(args)}) : ({', '.join(shapes)}) -> ({shapes[-1]})"

    
    idfs = __handling_identifiers(operation, full_code, args)

    # Handling maps
    maps_identifiers = re.findall(r"#(\w+)[^\w]",operation)

    maps_identifiers = set(maps_identifiers)

    maps = ""
    if maps_identifiers:
        lines = full_code.splitlines()
        for map_id in maps_identifiers:
            for line in lines:
                if f"{map_id} = affine_map" in line:
                    maps += line + '\n'
                    break
    
    #############################################################
    # All code:

    code = ""
    if maps is not None:
        code += f"{maps}\n"
    code += 'module attributes {torch.debug_module_name = "Net"} {\n'
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printFlops(f64)\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printNewline()\n"
    code += "func.func private @printMemrefF32(tensor<*xf32>)\n"
    code += f"{additional_function}\n" if additional_function else ""
    code += "\n"
    code += "\n"
    code +=f"func.func @matmul() -> {shapes[-1]}{{\n"
    code += "\n"
    code += "%val = arith.constant 2.00000e+00 : f32\n"
    code += "%zero = arith.constant 0.00000e+00 : f32\n"
    code += "\n"
    
    # code +=f"%out = bufferization.alloc_tensor() : tensor<{N}x{K}xf32>\n"
    # code +=f"%A = linalg.fill ins(%val : f32) outs(%out : tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32>\n"
    n_fill = 0
    for arg, shape, arg_dims in zip(args, shapes, dims):
        # print_info(arg,shape,arg_dims)
        if shape != 'f32':
            tmp_arg = f'%tmp_{arg[1:]}'
            code +=f"{tmp_arg} = bufferization.alloc_tensor() : {shape}\n"
            code +=f"{arg} = linalg.fill ins(%val : f32) outs({tmp_arg} : {shape}) -> {shape}\n"
            n_fill += 1
        else:
            code +=f"{arg} = arith.constant 2.00000e+00 : f32\n"
    
    code += "\n"
    code += "\n".join(idfs) + "\n" if idfs else ''
    code += "%t0 = func.call @nanoTime() : () -> (i64)\n"
    code += "\n"
    
    # code +=f"%D = linalg.matmul ins(%A, %B: tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%C: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>\n"
    code += f"%return_arg = {operation}"
    
    code += "\n"
    code += "%t = func.call @nanoTime() : () -> (i64)\n"
    code += "%delta = arith.subi %t, %t0 : i64\n"
    code += "%fp = arith.uitofp %delta : i64 to f64\n"
    code += "// func.call @printFlops(%fp) : (f64) -> ()\n"
    code += "func.call @printI64(%delta) : (i64) -> ()\n"
    code += "func.call @printNewline() : () -> ()\n"
    code += "\n"
    code +=f"return %return_arg : {shapes[-1]} \n"
    code += "}\n"
    code += "\n"
    code += "func.func @main(){\n"
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "    scf.for %i = %c0 to %n step %c1 {\n"
    code +=f"    %outputmain = func.call @matmul() : () -> {shapes[-1]}\n"
    code += "    }\n"
    code += "    return\n"
    code += "}\n"
    code += "}\n"

    return code,n_fill

def __inline(code: str, tmp_file_path: str):
    # Write the MLIR code to a temporary file
    with open(tmp_file_path, "w") as file:
        file.write(code)

    # Lower the Linalg dialect code to Affine dialect
    out = os.popen(f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt --inline {tmp_file_path}").read()

    if out != '':
        return out
    else:
        return None

def __handling_identifiers(operation: str, full_code: str, args):

    idf_defined_elsewhere = re.findall(r"(%\w+)[^\w](?!\s*=)",operation) # matches %15 but not %14 =

    idf_defined_inside = re.findall(r"(%\w+)[^\w]=",operation)

    idf_identifiers = set(idf_defined_elsewhere) - (set(idf_defined_inside).union(set(args)))

    if not args and "for" in operation:
        idf_identifiers = [idf for idf in idf_identifiers if "for" not in __getLine(operation, idf)]
        line = operation.splitlines()[0]
        idf_identifiers.extend(re.findall(r'outs\s*\(%\w+\s*=\s*(%\w+)\)',line))    

    # print(idf_identifiers)

    idfs = []
    if idf_identifiers:

        lines = full_code.splitlines()
        for idf in idf_identifiers:

            for line in lines:
                if f"{idf} =" in line:                 

                    if line.strip() not in operation:
                        if "{" in line and line.count("{") != line.count("}"):
                            word = re.findall(r"([a-zA-Z][^\s]+)",line.strip())[0]
                            line = extract_function(full_code, word)

                        line_idfs = __handling_identifiers(line, full_code, set())
                        if line_idfs:
                            idfs.extend(line_idfs)
                    
                    idfs.append(line.strip())
                    break

        idfs_set = set()
        final_idfs = []
        for idf in idfs:
            if not idf in idfs_set:
                idfs_set.add(idf)
                final_idfs.append(idf)

        idfs = final_idfs
    
    return idfs

def __getLine(text: str, word: str):
    for line in text.splitlines():
        if word in line:
            return line
        
    return ""

def __getArgsShapes(operation, additional_function=None):
    ins_outs_pattern = "(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    if fields == [] and additional_function is not None:
	# # TODO: Add shape extraction so that allocation snippet could be replicated
        fields = re.findall("(?:\(([^(]+)\))(?:\s*\->\s*([^(]+))", operation)[0]
        
        args,shapes = [],[]
        for f in fields[0].split(","):
            shapes.append(f.strip())
        shapes.append(fields[1])

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

    return args, shapes

def transform_wrapper_3(operation:str, full_code:str, additional_function: Optional[str] = None):

    args, shapes = __getArgsShapes(operation, additional_function)

    idfs, idfs_shapes = __handling_identifiers_3(operation, full_code, args)

    if idfs:
        args = idfs + args
        shapes = idfs_shapes + shapes
        args, shapes = __remove_duplicate_args(args, shapes)
    
    # print(args,shapes)
    
    #############################################################
    # consts:
    dims = []
    unique_dims = set()
    for shape in shapes:
        if shape.startswith("tensor"):
            arg_dims = list(map(int, re.findall(r'\d+', shape[7:-5])))
            dims.append( arg_dims )
            unique_dims = unique_dims.union(arg_dims)
        else: # shape == "f32"
            dims.append( -1 )
            unique_dims = unique_dims.union([-1])

    unique_dims = sorted(list(unique_dims))

    # Handling maps
    maps_identifiers = re.findall(r"#(\w+)[^\w]",operation)

    maps_identifiers = set(maps_identifiers)

    maps = ""
    if maps_identifiers:
        lines = full_code.splitlines()
        for map_id in maps_identifiers:
            for line in lines:
                if f"{map_id} = affine_map" in line:
                    maps += line + '\n'
                    break
    
    #############################################################
    # All code:

    code = ""
    if maps is not None:
        code += f"{maps}\n"
    code += 'module attributes {torch.debug_module_name = "Net"} {\n'
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printFlops(f64)\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printNewline()\n"
    code += "func.func private @printMemrefF32(tensor<*xf32>)\n"
    code += f"{additional_function}\n" if additional_function else ""
    code += "\n"
    code += "\n"
    code +=f"func.func @matmul() -> {shapes[-1]}{{\n"
    code += "\n"
    code += "%val = arith.constant 2.00000e+00 : f32\n"
    code += "%zero = arith.constant 0.00000e+00 : f32\n"
    code += "\n"
    
    # code +=f"%out = bufferization.alloc_tensor() : tensor<{N}x{K}xf32>\n"
    # code +=f"%A = linalg.fill ins(%val : f32) outs(%out : tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32>\n"
    n_fill = 0
    for arg, shape, arg_dims in zip(args, shapes, dims):
        # print_info(arg,shape,arg_dims)
        if shape not in ['f32','index']:
            tmp_arg = f'%tmp_{arg[1:]}'
            code +=f"{tmp_arg} = bufferization.alloc_tensor() : {shape}\n"
            code +=f"{arg} = linalg.fill ins(%val : f32) outs({tmp_arg} : {shape}) -> {shape}\n"
            n_fill += 1
        elif shape == "index":
            code += f"{arg} = arith.constant 1 : index \n"
        else:
            code +=f"{arg} = arith.constant 2.00000e+00 : f32\n"
    
    code += "\n"
    code += "%t0 = func.call @nanoTime() : () -> (i64)\n"
    code += "\n"
    
    # code +=f"%D = linalg.matmul ins(%A, %B: tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%C: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>\n"
    code += f"%return_arg = {operation}"
    
    code += "\n"
    code += "%t = func.call @nanoTime() : () -> (i64)\n"
    code += "%delta = arith.subi %t, %t0 : i64\n"
    code += "%fp = arith.uitofp %delta : i64 to f64\n"
    code += "// func.call @printFlops(%fp) : (f64) -> ()\n"
    code += "func.call @printI64(%delta) : (i64) -> ()\n"
    code += "func.call @printNewline() : () -> ()\n"
    code += "\n"
    code +=f"return %return_arg : {shapes[-1]} \n"
    code += "}\n"
    code += "\n"
    code += "func.func @main(){\n"
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "    scf.for %i = %c0 to %n step %c1 {\n"
    code +=f"    %outputmain = func.call @matmul() : () -> {shapes[-1]}\n"
    code += "    }\n"
    code += "    return\n"
    code += "}\n"
    code += "}\n"

    return code,n_fill

def __handling_identifiers_2(operation: str, full_code: str, args):

    idf_defined_elsewhere = re.findall(r"(%\w+)[^\w](?!\s*=)",operation) # matches %15 but not %14 =

    idf_defined_inside = re.findall(r"(%\w+)[^\w]=",operation)

    idf_identifiers = set(idf_defined_elsewhere) - (set(idf_defined_inside).union(set(args)))

    if not args and "for" in operation:
        idf_identifiers = [idf for idf in idf_identifiers if "for" not in __getLine(operation, idf)]
        line = operation.splitlines()[0]
        idf_identifiers.extend(re.findall(r'outs\s*\(%\w+\s*=\s*(%\w+)\)',line))    

        # print(idf_identifiers)

    idfs = []
    shapes = []
    if idf_identifiers:

        lines = full_code.splitlines()
        lines = [line.strip() for line in lines]
        for idf in idf_identifiers:
            if idf.startswith("%arg"):
                for i,v in enumerate(lines):
                    if operation.splitlines()[0].strip() in v:
                        break
                
                found = False
                for line in lines[i::-1]:
                    if re.search(fr"scf.forall \([%\w+\s*,]*{idf}",line) is not None:
                        idfs.append(idf)
                        shapes.append("index")
                        found = True
                        break

                if found:
                    continue



            for line in lines:
                if f"{idf} =" in line:                 

                    # if line.strip() not in operation:
                    #     if "{" in line and line.count("{") != line.count("}"):
                    #         # BUG imagine two scf.forall (sol: think about extracting the entire line? | using a curr index to start with)
                    #         word = re.findall(r"([a-zA-Z][^\s]+)",line.strip())[0]
                    #         line = extract_function(full_code, word)

                    if "->" in line:
                        line = line.splitlines()[0]
                        shape = __extract_shape(line.split("->")[-1])
                        shapes.append(shape)
                    elif ":" in line:
                        line = line.splitlines()[0]
                        shape = __extract_shape(line.split(":")[-1])
                        shapes.append(shape)

                    idfs.append(idf)
                    
                    break

        idfs_set = set()
        final_idfs, final_shapes = [],[]
        for idf,shape in zip(idfs,shapes):
            if not idf in idfs_set:
                idfs_set.add(idf)
                final_idfs.append(idf)
                final_shapes.append(shape)

        idfs,shapes = final_idfs, final_shapes
    
    return idfs, shapes

def __handling_identifiers_3(operation: str, full_code: str, args):

    idf_defined_elsewhere = re.findall(r"(%\w+)[^\w](?!\s*=)",operation) # matches %15 but not %14 =

    idf_defined_inside = re.findall(r"(%\w+)[^\w]=",operation)

    idf_identifiers = set(idf_defined_elsewhere) - (set(idf_defined_inside).union(set(args)))

    if not args and "for" in operation:
        idf_identifiers = [idf for idf in idf_identifiers if "for" not in __getLine(operation, idf)]
        line = operation.splitlines()[0]
        idf_identifiers.extend(re.findall(r'outs\s*\(%\w+\s*=\s*(%\w+)\)',line))    

        # print(idf_identifiers)

    idfs = []
    shapes = []
    if idf_identifiers:

        lines = full_code.splitlines()
        lines = [line.strip() for line in lines]
        for i,v in enumerate(lines):
            if operation.splitlines()[0].strip() in v:
                break

            match = re.search(r"{tag = \"\w+\"}",operation)
            if match is not None:
                if match.group(0) in v:
                    break


        linelist = lines[i::-1] if i != len(lines) - 1 else lines

        if i == len(lines) - 1:
            print("",end="")

        for idf in idf_identifiers:
            if idf.startswith("%arg"):

                found = False
                for line in linelist:
                    if re.search(fr"scf.forall \([%\w+\s*,]*{idf}",line) is not None:
                        idfs.append(idf)
                        shapes.append("index")
                        found = True
                        break

                if found:
                    continue



            for line in linelist:
                if f"{idf} =" in line:                 

                    # if line.strip() not in operation:
                    #     if "{" in line and line.count("{") != line.count("}"):
                    #         # BUG imagine two scf.forall (sol: think about extracting the entire line? | using a curr index to start with)
                    #         word = re.findall(r"([a-zA-Z][^\s]+)",line.strip())[0]
                    #         line = extract_function(full_code, word)

                    if "->" in line:
                        line = line.splitlines()[0]
                        shape = __extract_shape(line.split("->")[-1])
                        shapes.append(shape)
                    elif ":" in line:
                        line = line.splitlines()[0]
                        shape = __extract_shape(line.split(":")[-1])
                        shapes.append(shape)

                    idfs.append(idf)
                    
                    break

        idfs_set = set()
        final_idfs, final_shapes = [],[]
        for idf,shape in zip(idfs,shapes):
            if not idf in idfs_set:
                idfs_set.add(idf)
                final_idfs.append(idf)
                final_shapes.append(shape)

        idfs,shapes = final_idfs, final_shapes
    
    return idfs, shapes

def __extract_shape(return_part: str) -> str:
    match = re.search(r"(tensor<(?:[0-9]+x)+f32>)|(f32)", return_part)
    if match is None:
        raise ValueError("could not find any shape")
    else:
        return match.group(0)