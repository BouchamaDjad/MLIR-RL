from dotenv import load_dotenv
load_dotenv(override=True)

from rl_autoschedular.observation import (
    extract_op_features_from_affine_code,
    __inline,
    main_wrapper
)

from rl_autoschedular.evaluation import evaluate_code_with_bindings_and_timeout

from tqdm import tqdm
import json
import numpy as np

import yaml
from dataclasses import asdict
import traceback
import argparse
import random

from data_utils.generation import (
    LINALG_OPERATION_GENERATORS,
    BATCH_SIZES,
    HEIGHTS,
    CHANNELS,
    KERNELS,
    DILATIONS,
    STRIDES,
    SIZES,
    randomSubGraph,
    randomblocks,
    # ResNetBlock,
    generate_resnet_block,
    generate_residual_block_mlir
)

tmp_file = 'tmp/tmp-3.mlir'

class ParserMock:
    def __init__(self,input_file,output_file):
        self.input_file = input_file
        self.output_file = output_file


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process an input file and save to an output file.')
    parser.add_argument('--input_file', type=str, help='The input file to be processed.')
    parser.add_argument('--output_file', type=str, help='The file where the processed content will be saved.')

    args = parser.parse_args()

    print(args.input_file,args.output_file)

    # args = ParserMock(input_file="config/config.yaml",output_file="t.json")
    
    with open(args.input_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Set the shapes of the operations 
    BATCH_SIZES.extend(config['SHAPES']['BATCH_SIZES']) # Used by all the operations
    HEIGHTS.extend(config['SHAPES']['HEIGHTS']) # Used by operations on images
    CHANNELS.extend(config['SHAPES']['CHANNELS']) # Used by operations on images
    KERNELS.extend(config['SHAPES']['KERNELS']) # Used by operations on images
    DILATIONS.extend(config['SHAPES']['DILATIONS']) # Used by operations on images
    STRIDES.extend(config['SHAPES']['STRIDES']) # Used by operations on images
    SIZES.extend(config['SHAPES']['SIZES']) # Used on other operations like matmul, add, etc...    

    operations_config = {
        f"single_{operation_name}": (LINALG_OPERATION_GENERATORS[operation_name], amount) for operation_name, amount in config['OPERATIONS'].items() if amount > 0
    }

    print( sum( amount for _, (_, amount) in operations_config.items() ) )

    # operations_config.update({
    #     "bench": (randomSubGraph, 1400),

    #     "pattern-Linear-block (sigmoid)": (
    #         lambda :randomblocks(operations=[
    #             "matmul",
    #             "add",
    #             "sigmoid"
    #         ]),100
    #     ),

    #     "pattern-Linear-block (relu)": (
    #         lambda :randomblocks(operations=[
    #             "matmul",
    #             "add",
    #             "relu"
    #         ]),100
    #     ),

    #     "pattern-Conv2d-block": (
    #         lambda :randomblocks(operations=[
    #             "conv_2d_nchw_fchw",
    #             "relu"
    #         ]),100
    #     ),

    #     "pattern-Resnet": (generate_resnet_block, 100),

    #     "pattern-Residual-block": (generate_residual_block_mlir, 100)

    # })

    print(operations_config)

    all_operations = {}

    for operation_name, (generator, amount) in tqdm(operations_config.items(), desc="linalg operations"):

        # Iterate the specified number of times ('amount') for the current operation
        for i in tqdm(range(amount), desc=operation_name):
            
            exec_time = None  # Initialize execution time as None to enter the loop
            
            maps = None
            additional_function = None
            try:
                res = generator()  # Generate the raw operation using the provided generator function

                if isinstance(res, tuple):
                    raw_operation, additional_tuple = res
                    if isinstance(additional_tuple, tuple):
                        maps, additional_function = additional_tuple
                    else:
                        maps = additional_tuple
                else:
                    raw_operation = res

                print(raw_operation)
                
                # Get operation features
                # op_features = extract_op_features_from_affine_code(raw_operation,tmp_file,maps,additional_function)
                
                # loops_data = asdict(op_features)  # Convert the dataclass to a dictionary
                # loops_data.pop("raw_operation")  # Remove raw_operation
                
                
                transform_wrapped_operation = main_wrapper(raw_operation, tmp_file ,maps=maps, additional_function=additional_function)
                # transform_wrapped_operation = inline(transform_wrapped_operation, tmp_file)

                # Evaluate the execution time of the transformed operation with a timeout of 300 seconds
                exec_time, assertion = evaluate_code_with_bindings_and_timeout(transform_wrapped_operation, 300)

                # If the execution time is valid, calculate a more stable median execution time
                if assertion and exec_time is not None:
                    exec_time = np.median([exec_time] + [evaluate_code_with_bindings_and_timeout(transform_wrapped_operation, 300)[0] for _ in range(2)])
            
            except Exception as e:
                print(f"\033[91;1mError occurred while generating operation: {e}\033[0m") 
                traceback.print_exc()
                continue
            
            # If a valid execution time was obtained, store the operation details
            if exec_time:
                
                # all_operations[f"{operation_name}_{i}"] = {
                all_operations[f"single_{operation_name}_{i}"] = {
                    "operation": raw_operation,  # The raw operation
                    "transform_wrapped_operation": transform_wrapped_operation,  # The transformed wrapped operation
                    # "loops_data": loops_data,  # Data related to the loops in the operation
                    "execution_time": exec_time,  # The median execution time
                }
            else:
                continue  # If no valid execution time, skip to the next iteration
        
    
    # Remove duplicates from all_operations based on 'transform_wrapped_operation' value
    unique_operations = {}
    unique_transforms_wrapped = set()
    
    for key, value in all_operations.items():
        if value["transform_wrapped_operation"] not in unique_transforms_wrapped:
            unique_transforms_wrapped.add(value["transform_wrapped_operation"])
            unique_operations[key] = value

    print(len(all_operations))
    del all_operations # To save memory

    with open(args.output_file, 'w') as file:
        json.dump(unique_operations, file)

    # with open(args.output_file, 'w') as file:
    #     json.dump(all_operations, file)