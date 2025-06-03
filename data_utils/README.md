## Guide to generate datasets using data_generation_* scripts
- set `PYTHONPATH` environment variable to inlcude the project (MLIR-RL) directory (
    otherwise it fails  `ImportError: attempted relative import`
)

## Guide To generate mlir code from Neural Networks impelemented in pytorch
1. Need to install [torch-mlir](https://github.com/llvm/torch-mlir)

1. use the torch-mlir-generation.py and then the main_wrapper.py

1. the resulting mlir file should be usable  in evaluate.py