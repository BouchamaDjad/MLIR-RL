from rl_autoschedular.transforms import (
    transform_dialect_fusion,
    transform_dialect_tile,
    transform_dialect_fuse_only,
    transform_dialect_vectorise,
    transform_dialect_interchange,
    transform_dialect_TP,
    transform_dialect_vectorise_with_vectorizer
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

# with open('./errors_files/11_operation_3.mlir', 'r') as file:
#     code = file.read()

# old_code = code
    
# old_exec, bench_passed = evaluate_code_with_cmd_and_timeout(code, tmp_file)
    
a = json.load(open(file_path)) 
code = a[list(a.keys())[10]]['transform_wrapped_operation']
old_exec = a[list(a.keys())[10]]['execution_time']

bench = extract_bench_features_from_code('bench_1',code, 60, 60)

code = bench.code

# op_features = bench.operations['operation_4']



#observations:

'''
   vectorisation alone doesnt make sens
   parall + vect works (if the producer of the operation is a fill again whyyyy?)
   vectorisation seems to work only if all ops except fill operations are fused together the producer of the current operation is a fill

'''

# rules :
'''
  ops: 
     - pooling:
            vectorisation doesnt work with pooling

  for op X , if i fuse or TP -> if it has a fill as a prodcuer -|> fuse the fill with op X and vectorise X and the fill operation, if fusion, also vectorise the consumer
 
  for op X if i fuse 
 Tiling -> no vectorisation possible
 vectorise after fusion only if producer is a fill
 vectorisation -|> if its the first producer in the chain and its inside an scf.for
                   
 
 
 we can TP the first linalg operation + fuse (only) the fill operations and then vectorise, is it always good? (possibly)
 
 if i tp the first producer and tp the consumer after wards we cant fuse anymore (both are tiled) (if both exist in the tiled list mask fusion)
    - but we can still vectorise it
    - fusing then vectorising is way superior than TP alone
 
'''

# code = transform_dialect_TP(code,'operation_3',[2,2],op_features.nested_loops,tmp_file)
# op_features = bench.operations['operation_4']

# code = transform_dialect_TP(code,'operation_4',[2,2],op_features.nested_loops,tmp_file)

# code = transform_dialect_tile(code,'operation_3',[2,2],tmp_file)
# code = transform_dialect_fuse_only(code,'operation_4','operation_3', tmp_file)
# code = transform_dialect_interchange(code,'operation_3',[1,0,2],tmp_file)
 
code = transform_dialect_fusion(code,'operation_4','operation_3',[2,2], tmp_file)

# code = transform_dialect_fuse_only(code,'operation_3','operation_2', tmp_file)
# code = transform_dialect_fuse_only(code,'operation_3','operation_1', tmp_file)
# code = transform_dialect_fuse_only(code,'operation_3','operation_0', tmp_file)

# code = transform_dialect_vectorise_with_vectorizer(code,'operation_4',tmp_file)

# code = transform_dialect_fuse_only(code,'operation_5','operation_4', tmp_file)
# code = transform_dialect_vectorise_with_vectorizer(code,'operation_5',tmp_file)
code = transform_dialect_vectorise_with_vectorizer(code,'operation_4',tmp_file)

# code = transform_dialect_vectorise_with_vectorizer(code,'operation_5',tmp_file)
# code = transform_dialect_vectorise_with_vectorizer(code,'operation_3',tmp_file)

# code = transform_dialect_fuse_only(code,'operation_3','operation_2', tmp_file)
# code = transform_dialect_fuse_only(code,'operation_3','operation_1', tmp_file)
# code = transform_dialect_fuse_only(code,'operation_3','operation_0', tmp_file)

# code = transform_dialect_vectorise_with_vectorizer(code,'operation_3',tmp_file)

# code = transform_dialect_vectorise_with_vectorizer(code,'operation_2',tmp_file)

# code = transform_dialect_vectorise_with_vectorizer(code,'operation_1',tmp_file)
# code = transform_dialect_vectorise_with_vectorizer(code,'operation_0',tmp_file)



# code = transform_dialect_vectorise_with_vectorizer(code,'operation_6',tmp_file)



# code = transform_dialect_vectorise(code,'operation_3',tmp_file)

with open('./result_fusion.mlir', 'w') as file:
    file.write(code)

print('--- fusion + vect ---')
new_exec_time, bench_passed = evaluate_code_with_cmd_and_timeout(code, tmp_file, 120)

# # # old_exec, bench_passed = evaluate_code_with_cmd_and_timeout(old_code, tmp_file)


print(bench_passed)
print(new_exec_time)
print('speed up :', old_exec/new_exec_time)
