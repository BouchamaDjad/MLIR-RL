import torch
import torchvision.models as models

from torch_mlir.fx import export_and_import
from torch_mlir.compiler_utils import lower_mlir_module, OutputType

model = models.resnet18(weights="DEFAULT").eval()
# model = models.mobilenet_v2(weights='DEFAULT').eval()
# model = models.vgg.vgg11(weights="DEFAULT").eval()

model_name = model.__class__.__name__
print(model_name)

module = export_and_import(
    model,
    torch.ones(1, 3, 224, 224),
    output_type="linalg-on-tensors",
    func_name="forward"
)



print("------------------------ Loaded Resnet18 into Module ---------------------------------")

# Convert to Torch dialect
# mlir_module_torch = lower_mlir_module(module=module, output_type=OutputType("torch"), verbose=False)

# # Convert to TOSA dialect
# mlir_module_tosa = lower_mlir_module(module=module, output_type=OutputType("tosa"), verbose=False)

# Convert to Linalg-on-Tensors dialect
mlir_module_linalg = lower_mlir_module(module=module, output_type=OutputType("linalg-on-tensors"), verbose=False)

print("------------------------ Converted Resnet18 into MLIR ---------------------------------")

# with open(f"{model_name}_torch.mlir", "w") as f:
#     f.write(str(mlir_module_torch))

# with open(f"{model_name}_linalg_asm.mlir","w") as f:
#     f.write(mlir_module_linalg.operation.get_asm(enable_debug_info=False, binary=False))

with open(f"{model_name}_linalg.mlir", "w") as f:
    f.write(str(mlir_module_linalg))

# with open("resnet18_tosa.mlir", "w") as f:
#     f.write(str(mlir_module_tosa))


print("------------------------ Saved Resnet18 MLIR ---------------------------------")
