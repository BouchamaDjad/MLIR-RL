import torch
# import intel_extension_for_pytorch as ipex
# ipex.enable_onednn_fusion(True)
import time
from torchvision import models
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GraphSAGE
from torch_geometric.datasets import Planetoid


"""Python Script to measure execution times of multiple NN models with pytorch optimizations"""

# Select device
device = torch.device("cpu")

def measure_time(func, num_trials=10):
    # Warm-up run
    for _ in range(10):
        func()
    
    # Measure time
    start = time.time()
    for _ in range(num_trials):
        func()
    
    end = time.time()
    
    return (end - start) / num_trials

pubmed_dataset = Planetoid(root="data", name="Pubmed")

def extract_multikeys_values(data: dict[str,any], keys: list[str]) -> tuple:
    return tuple(
        [v for k,v in data.items() if k in keys]
    )

# Load a sample model (ResNet18)
bench_models = [
    # (models.resnet50(weights="DEFAULT").to(device).eval(),torch.randn(1, 3, 224, 224).to(device)),
    # (models.mobilenet_v2(weights="DEFAULT").to(device).eval(),torch.randn(1, 3, 224, 224).to(device)),
    # (models.vgg16(weights="DEFAULT").to(device).eval(),torch.randn(1, 3, 224, 224).to(device))
    # (models.convnext_tiny(weights="DEFAULT").eval(), torch.randn(1, 3, 224, 224)),
    # (models.efficientnet_b3(weights="DEFAULT").eval(), torch.randn(1, 3, 224, 224)),
    # (AutoModel.from_pretrained("bert-base-uncased", torchscript=True).eval(), extract_multikeys_values(AutoTokenizer.from_pretrained("bert-base-uncased")("Hello from MLIR", return_tensors="pt", padding="max_length", max_length=16, truncation=True), keys=["input_ids","attention_mask"])),
    # (AutoModel.from_pretrained("distilbert-base-uncased", torchscript=True).eval(), extract_multikeys_values(AutoTokenizer.from_pretrained("distilbert-base-uncased")("Hello from MLIR", return_tensors="pt", padding="max_length", max_length=16, truncation=True), keys=["input_ids","attention_mask"])),
    (GraphSAGE(in_channels=pubmed_dataset.num_features, hidden_channels=64, num_layers=2, out_channels=pubmed_dataset.num_classes, dropout=0.5), (pubmed_dataset[0].x, pubmed_dataset[0].edge_index)),

]


def main():
    # Create operations
    print("Measuring execution times (in seconds)...")
    
    # Measure execution times for Base and JIT
    print(f"{'Model':<40} {'Base':<15} {'JIT':<15}")

    with open("model-eval-pytorch-1.csv","a") as f:   
        f.write("Name,base-torch,torch-jit\n") 
        for model,input in bench_models:
            print("-" * 70)

            name = model.__class__.__name__

            base_time = measure_time(lambda :model(input) if not isinstance(input,tuple) else model(*input))

            model = torch.jit.script(model)

            jit_time = measure_time(lambda :model(input) if not isinstance(input,tuple) else model(*input))
        
            print(f"{name:<40} {base_time:<15.6f} {jit_time:<15.6f}")
            f.write(f"{name:<20},{base_time},{jit_time}\n")


if __name__ == "__main__":
    main()