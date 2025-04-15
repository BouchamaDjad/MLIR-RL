from dotenv import load_dotenv

from utils.log import print_info
load_dotenv(override=True)

from rl_autoschedular.observation import (
    extract_op_features_from_affine_code,
    __transform_wrapper
)
from rl_autoschedular.evaluation import evaluate_code_with_cmd_and_timeout
from random import randint, choice, shuffle, random
from tqdm import tqdm
import json
import numpy as np
import yaml
from dataclasses import asdict
import argparse
from utils.generation import LINALG_OPERATION_GENERATORS,BATCH_SIZES,HEIGHTS,CHANNELS,KERNELS,DILATIONS,STRIDES,SIZES,randomSubGraph

tmp_file = 'tmp/temp_mlir.mlir'

class ParserMock:
    def __init__(self,input_file,output_file):
        self.input_file = input_file
        self.output_file = output_file


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process an input file and save to an output file.')
    parser.add_argument('--input_file', type=str, help='The input file to be processed.')
    parser.add_argument('--output_file', type=str, help='The file where the processed content will be saved.')

    args = parser.parse_args()

    # print_info(args.input_file,args.output_file)

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

    # operations_config = {
    #     operation_name: (LINALG_OPERATION_GENERATORS[operation_name], amount) for operation_name, amount in config['OPERATIONS'].items() if amount > 0
    # }

    operations_config = {
        "randomSubGraph": (randomSubGraph, 300)
    }

    # print( sum( amount for _, (_, amount) in operations_config.items() ) )

    all_operations = {}

    for operation_name, (generator, amount) in tqdm(operations_config.items(), desc="linalg operations"):

        # Iterate the specified number of times ('amount') for the current operation
        for i in tqdm(range(amount), desc=operation_name):
            
            exec_time = None  # Initialize execution time as None to enter the loop
                
            # Loop until a valid execution time is obtained
            # while exec_time is None:
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
                
                # Get operation features

                print(raw_operation)
                
                op_features = extract_op_features_from_affine_code(raw_operation,tmp_file,maps,additional_function)
                
                loops_data = asdict(op_features)  # Convert the dataclass to a dictionary
                loops_data.pop("raw_operation")  # Remove raw_operation
                
                
                transform_wrapped_operation = __transform_wrapper(raw_operation, maps=maps, additional_function=additional_function)

                # Evaluate the execution time of the transformed operation with a timeout of 300 seconds
                exec_time, assertion = evaluate_code_with_cmd_and_timeout(transform_wrapped_operation, tmp_file, 300)

                # If the execution time is valid and below a certain threshold, calculate a more stable median execution time
                if assertion and exec_time < 1000000:
                    exec_time = np.median([exec_time] + [evaluate_code_with_cmd_and_timeout(transform_wrapped_operation, tmp_file,300)[0] for _ in range(2)])
            
            except Exception as e:
                print(f"\033[91;1mError occurred while generating operation: {e}\033[0m")   
                continue
            
            # If a valid execution time was obtained, store the operation details
            if exec_time:
                # print("write in progress")
                
                all_operations[f"{raw_operation}"] = {
                    "operation": raw_operation,  # The raw operation
                    "transform_wrapped_operation": transform_wrapped_operation,  # The transformed wrapped operation
                    "loops_data": loops_data,  # Data related to the loops in the operation
                    "execution_time": exec_time,  # The median execution time
                }
            else:
                continue  # If no valid execution time, skip to the next iteration
        
        # Write all the collected operation data to the output file in JSON format
    with open(args.output_file, 'w') as file:
        json.dump(all_operations, file)
