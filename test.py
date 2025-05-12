from rl_autoschedular.transforms import (
    transform_dialect_fusion,
    transform_dialect_tile,
    transform_dialect_fuse_only,
    transform_dialect_vectorise,
    transform_dialect_interchange,
    transform_dialect_TP
)

from rl_autoschedular.env import (
    fix
)

import json

from rl_autoschedular.evaluation import evaluate_transform_with_cmd, evaluate_code_with_cmd_and_timeout, evaluate_code_with_cmd
from rl_autoschedular.observation import __function_wrapper, __lower_linalg_to_loops, get_raw_ast_info, extract_bench_features_from_code

tmp_file = 'tmp/test_fusion.mlir'
tmp_file_2 = 'tmp/before_fusion.mlir'

file_path = './data/nn/operations-sequence.json'

with open('./test_code.mlir', 'r') as file:
    code = file.read()
    
old_exec, bench_passed = evaluate_code_with_cmd_and_timeout(code, tmp_file)
    
# a = json.load(open(file_path))
# code = a[list(a.keys())[0]]['transform_wrapped_operation']
# old_exec = a[list(a.keys())[0]]['execution_time']

bench = extract_bench_features_from_code('bench_1',code, 60, 60)

code = bench.code

op_features = bench.operations['operation_5']


code = transform_dialect_TP(code,'operation_5',[2, 2, 2],op_features.nested_loops,tmp_file)
# code = transform_dialect_tile(code,'operation_4',[2,2],tmp_file)
# code = transform_dialect_interchange(code,'operation_4',[1,0],tmp_file)
 
# code = transform_dialect_fusion(code,'operation_4','operation_3',[2,2], tmp_file)
# code = transform_dialect_fuse_only(code,'operation_3','operation_2', tmp_file)
# code = transform_dialect_vectorise(code,'operation_4',tmp_file)
# code = transform_dialect_vectorise(code,'operation_3',tmp_file)

new_exec_time, bench_passed = evaluate_code_with_cmd_and_timeout(code, tmp_file)

print(bench_passed)
print(new_exec_time)
print('speed up :', old_exec/new_exec_time)
with open('./result_fusion.mlir', 'w') as file:
    file.write(code)