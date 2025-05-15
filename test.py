from rl_autoschedular.transforms import (
    transform_dialect_TP,
    transform_dialect_fusion,
    transform_dialect_tile,
    transform_dialect_fuse_only,
    transform_dialect_vectorise,
    transform_dialect_vectorise_with_vectorizer
)

from rl_autoschedular.env import (
    fix
)

import json
import numpy as np

from rl_autoschedular.evaluation import evaluate_transform_with_cmd, evaluate_code_with_cmd_and_timeout
from rl_autoschedular.observation import __function_wrapper, __lower_linalg_to_loops, extract_bench_features_from_code

code_file = 'tmp/test_fusion.mlir'
tmp_file = 'tmp.mlir'

with open(code_file, 'r') as f:
    code = f.read()

# data = json.load(open("final-operation-sequence-4.json"))
# data = data[list(data.keys())[0]]
# code = data["transform_wrapped_operation"]

#############################
# #BEGIN_GRAPH
# operation_0 --> operation_3
# operation_1 --> operation_3
# operation_2 --> operation_3
# operation_3 --> operation_4
# #END_GRAPH
#############################
# fils = {0, 1, 2}
# matmul = 3
# sigmoid = 4

# old_exec_time = np.median([evaluate_code_with_cmd_and_timeout(code, tmp_file) for _ in range(3)])
# old_exec_time = 18654385333.0
# old_exec_time = 37293085760
old_exec_time = 18646470475.0
# old_exec_time = data["execution_time"]
print("old_exec_time", old_exec_time)

bench_features = extract_bench_features_from_code("bench", code, old_exec_time, old_exec_time)

print(bench_features.operations)

# print(bench_features.operations["operation_5"])

# code = transform_dialect_TP(code, "operation_3", [2,2], bench_features.operations["operation_3"].nested_loops ,tmp_file)
# code = transform_dialect_tile(code, "operation_3", [2,2], tmp_file)
# code = transform_dialect_fusion(code, "operation_4", "operation_3", [2,2] ,tmp_file)
# code = transform_dialect_fuse_only(code,'operation_5','operation_3', tmp_file)

code = transform_dialect_fuse_only(code, 'operation_3','operation_2', tmp_file)
code = transform_dialect_fuse_only(code, 'operation_3','operation_1', tmp_file)
code = transform_dialect_fuse_only(code, 'operation_3','operation_0', tmp_file)


code = transform_dialect_vectorise_with_vectorizer(code, "operation_4", tmp_file)
code = transform_dialect_vectorise_with_vectorizer(code, "operation_3", tmp_file)
# code = transform_dialect_vectorise(code, "operation_3", tmp_file)

code = transform_dialect_vectorise_with_vectorizer(code, "operation_2", tmp_file)
code = transform_dialect_vectorise_with_vectorizer(code, "operation_1", tmp_file)
code = transform_dialect_vectorise_with_vectorizer(code, "operation_0", tmp_file)

# op_iter_space = 1
# for nested_loop in bench_features.operations["operation_3"].nested_loops:
#     op_iter_space *= nested_loop.upper_bound
    
# print(f"REASON: Too large to vectorize {op_iter_space}")

print(code != "" , code is not None)

if code is not None and code != "":
    new_exec_time, passed = evaluate_code_with_cmd_and_timeout(code, tmp_file)

    if passed:
        print("New execution time:", new_exec_time)
        print("Speedup:", old_exec_time / new_exec_time)
    else:
        print(new_exec_time)
else:
    print("Code is None or empty")
