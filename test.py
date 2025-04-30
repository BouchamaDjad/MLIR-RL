from rl_autoschedular.transforms import (
    transform_dialect_fusion
)

import json

from rl_autoschedular.evaluation import evaluate_transform_with_cmd, evaluate_code_with_cmd_and_timeout
from rl_autoschedular.observation import __function_wrapper, __lower_linalg_to_loops

tmp_file = 'tmp/test_fusion.mlir'
tmp_file_2 = 'tmp/before_fusion.mlir'

with open('./full_code_with_tags.mlir', 'r') as file:
    code = file.read()

code = transform_dialect_fusion(code,'operation_6','whatever', tmp_file)

result = evaluate_transform_with_cmd(code, tmp_file)

with open('./result_fusion.mlir', 'w') as file:
    file.write(result)