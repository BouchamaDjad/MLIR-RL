from dataclasses import dataclass
from typing import Literal
import numpy as np
from copy import deepcopy

class LoopNode:
    def __init__(self, arg, upper,lower, parent,vector):
        self.arg = arg
        self.upper = upper
        self.lower = lower
        self.children = []
        self.parent = parent
        self.vector = None
        self.instructions = []

    def __repr__(self, level=0):
        indent = "  " * level
        result = f"{indent}- {self.arg}\n"
        for child in self.children:
            result += child.__repr__(level + 1)
        return result

@dataclass
class NestedLoopFeatures:
    """Dataclass to store the nested loops features data."""
    arg: str
    """The argument representing the loop iterator."""
    lower_bound: int
    """The lower bound of the loop."""
    upper_bound: int
    """The upper bound of the loop."""
    step: int
    """The loop step."""
    iterator_type: Literal["parallel", "reduction"]
    """The type of the loop iterator."""


@dataclass
class OperationFeatures:
    """Dataclass to store the operation features data."""
    raw_operation: str
    """The raw operation string without wrapping or transformations."""
    op_count: dict[str, int]
    """Number of arithmetic operations in the operation."""
    load_data: list[list[str]]
    """List of load accesses where each load is represented by the list of access arguments."""
    store_data: list[list[str]]
    """List of store accesses where each store is represented by the list of access arguments."""
    nested_loops: list[NestedLoopFeatures]
    """List of nested loops where each loop is represented by the NestedLoopFeatures dataclass."""
    producers: list[str]
    """List of producers for an operation"""
    consumers: list[str]
    """List of consumers for an operation"""


@dataclass
class BenchmarkFeatures:
    """Dataclass to store the benchmark features data."""
    bench_name: str
    """The benchmark's name."""
    code: str
    """The MLIR code of the benchmark."""
    operation_tags: list[str]
    """List of operation tags."""
    operations: dict[str, OperationFeatures]
    """List of operations where each operation is represented by the OperationFeatures dataclass."""
    exec_time: int
    """Execution time of the benchmark in nanoseconds."""
    root_exec_time: int
    """Execution time of the benchmark in nanoseconds without any transformation."""

    def copy(self):
        """Create a deep copy of the BenchmarkFeatures instance."""
        return BenchmarkFeatures(
            bench_name=self.bench_name,
            code=self.code,
            operation_tags=deepcopy(self.operation_tags),
            operations=deepcopy(self.operations),
            exec_time=self.exec_time,
            root_exec_time=self.root_exec_time
        )


@dataclass
class OperationState:
    bench_name: str
    """The benchmark's name."""
    operation_tag: str
    """Tag used to identify the operation in the MLIR code."""
    operation_index: int
    """the index of the current operation with respect to all the operations of the given code"""
    operation_type: str
    """The type of the operation (generic, matmul, conv2d, ...)."""
    operation_features: OperationFeatures
    """Features of the operation."""
    current_producer: int
    "index of the current consumer"
    producer_tag: str
    "tag of the consumer operation"
    producer_features: OperationFeatures
    "Features of the consumer operation"
    fused_ops: set
    "fused ops"
    transformed_code: str
    """The operation string with wrapping and transformations."""
    actions: np.ndarray
    """Action parameters for parallelization, tiling and interchange. The shape is (MAX_NUM_LOOPS, 3, truncate)."""
    actions_mask: np.ndarray
    """Mask for the actions. The shape is (5 + L + L + (L-1) + (L-2) + (L-3)) where L = MAX_NUM_LOOPS."""
    step_count: int
    """The current step in the list of transformations applied to the operation."""
    exec_time: int
    """Execution time of the operation in nanoseconds."""
    root_exec_time: int
    """Execution time of the operation in nanoseconds without any transformation."""
    empty_schedule: bool
    """is True if the model has not chosen to apply any transformation on any operation """
    transformation_history: list[tuple[str, list[int]]]
    """List of transformations with their parameters applied to the operation."""
    cummulative_reward: float
    """Cummulative reward of the operation."""
    tmp_file: str
    """Temporary file to store the MLIR code."""
    
@dataclass
class LoopFeatures:
    """Dataclass to store the operation features data."""
    op_count: dict[str, int]
    """Number of arithmetic operations in the operation."""
    load_data: list[list[str]]
    """List of load accesses where each load is represented by the list of access arguments."""
    store_data: list[list[str]]
    """List of store accesses where each store is represented by the list of access arguments."""
    nested_loops: list[NestedLoopFeatures]
    """List of nested loops where each loop is represented by the NestedLoopFeatures dataclass."""
    
    
@dataclass
class ObservationFeatures:
    """ Dataclass to store the observation informations needed to pass into the model """
    consumer_tree : LoopNode
    """Tree representation of the consumer operation (current)"""
    producer_tree : LoopNode
    """Tree representation of the producer operation"""
    action_mask : list[int]
    """action mask for the current state"""
    action_history: list[int]
    """action history for the current operation"""

    
    
    
    
    
