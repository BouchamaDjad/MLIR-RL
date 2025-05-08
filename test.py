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
from rl_autoschedular.observation import __function_wrapper, __lower_linalg_to_loops

tmp_file = 'tmp/test_fusion.mlir'
tmp_file_2 = 'tmp/before_fusion.mlir'

with open('./full_code_with_tags.mlir', 'r') as file:
    code = file.read()

code = transform_dialect_fusion(code,'operation_6','operation_5',[2], tmp_file)
code = transform_dialect_fusion(code,'operation_5','operation_4',[2],tmp_file)
result = evaluate_transform_with_cmd(code, tmp_file)

with open('./result_fusion.mlir', 'w') as file:
    file.write(result)