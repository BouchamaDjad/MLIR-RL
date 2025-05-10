from rl_autoschedular.transforms import (
    transform_dialect_fusion,
    transform_dialect_tile,
    transform_dialect_fuse_only,
    transform_dialect_vectorise
)

from rl_autoschedular.env import (
    fix
)

import json

from rl_autoschedular.evaluation import evaluate_transform_with_cmd, evaluate_code_with_cmd_and_timeout
from rl_autoschedular.observation import __function_wrapper, __lower_linalg_to_loops, get_raw_ast_info

tmp_file = 'tmp/test_fusion.mlir'
tmp_file_2 = 'tmp/before_fusion.mlir'

with open('./current_bench.mlir', 'r') as file:
    code = file.read()

# print(code)
# ast = get_raw_ast_info(code, tmp_file)
# print(ast)
# with open('ast.txt', 'w', encoding='utf-8') as file:
#     file.write(ast)

code = transform_dialect_fusion(code,'operation_3','operation_2',[4,2], tmp_file)
result = evaluate_transform_with_cmd(code, tmp_file)

with open('./result_fusion.mlir', 'w') as file:
    file.write(result)