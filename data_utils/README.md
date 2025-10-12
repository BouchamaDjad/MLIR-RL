## Guide to generate datasets using data_generation_* scripts
- there are two versions with bindings and without (recommended to use bindings)
- set `PYTHONPATH` environment variable to inlcude the project (MLIR-RL) directory (
    otherwise it fails  `ImportError: attempted relative import`
)

## Guide To generate mlir code from Neural Networks (Visions models) impelemented in pytorch
1. Need to install [torch-mlir](https://github.com/llvm/torch-mlir)

1. use the torch-mlir-generation.py and then the main_wrapper.py (be sure to change the call of the main function in the latter)

1. the resulting mlir file should be usable in evaluate.py

## Guide To generate mlir code from Neural Networks (Transformers) implemented in pytorch

1. Need to install hugging face transformer library

1. run models-to-onnx.py script

2. the resulting file *might* be usable in evaluate.py (Some issues may still appear -espcially math.erf errors-)