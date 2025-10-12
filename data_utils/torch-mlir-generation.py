import torch
import torchvision.models as models

from torch_mlir.fx import export_and_import
from torch_mlir.compiler_utils import lower_mlir_module, OutputType

# model = models.resnet18(weights="DEFAULT").eval()
# model = models.mobilenet_v2(weights='DEFAULT').eval()
# model = models.vgg.vgg11(weights="DEFAULT").eval()

# model = models.convnext_tiny(weights="DEFAULT").eval()
model = models.convnext_large(weights="DEFAULT").eval()
# model = models.efficientnet_b3(weights="DEFAULT").eval()

model_name = model.__class__.__name__
print(model_name)

module = export_and_import(
    model,
    torch.ones(1, 3, 224, 224), # example input
    output_type="linalg-on-tensors",
    func_name=model_name
)

# torch.ones(1, 3, 300, 300), EffecienNet


print(f"------------------------ Loaded {model_name} into Module ---------------------------------")

# Convert to Linalg-on-Tensors dialect
mlir_module_linalg = lower_mlir_module(module=module, output_type=OutputType("linalg-on-tensors"), verbose=False)

print(f"------------------------ Converted {model_name} into MLIR ---------------------------------")

# with open(f"{model_name}_linalg_asm.mlir","w") as f:
#     f.write(mlir_module_linalg.operation.get_asm(enable_debug_info=False, binary=False))

with open(f"./data_utils/tmp_models/{model_name}_linalg.mlir", "w") as f:
    f.write(str(mlir_module_linalg))

print(f"------------------------ Saved {model_name} MLIR ---------------------------------")
